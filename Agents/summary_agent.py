# Agents/summary_agent.py
#the summary agent is responisble for summarizing textbook chapter pdf as a part of the textbook pipeline.

#the required libraries are:
import os
import re  #regular expression library used for cleaning text
import fitz #used to open and read pdf files
import ollama
from collections import Counter, defaultdict #used for analyzing pdf components such as font sizes, and blocks

from Prompts.summary_prompt import SUMMARY_PROMPT 

MIN_SECTION_CHARS  = 40
MAX_CHUNK_CHARS    = 6000  # send only 6000 characters to the LLM at once, to prevent hallucinations
HEADING_CHAR_LIMIT = 500   # a font cluster with more chars than this is body, not headings



MONO_FONTS = (
    "courier", "mono", "code", "consolas", "inconsolata",
    "sourcecodepro", "jetbrains", "firacode", "droidmono",
    "lucidaconsole", "anonymous", "terminus",
) #fonts probably used for code blocks


class SummaryAgent:

    def __init__(self, model_name="qwen2.5:7b", base_data_dir="Data"):
        self.model_name    = model_name
        self.base_data_dir = base_data_dir

    #the main function used to call the agent
    def run(self, pdf_path, output_txt_path=None, max_pages=None):
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"PDF not found: {pdf_path}")
        #used to extract the file name ---> ai.pf  == ai
        pdf_name = os.path.splitext(os.path.basename(pdf_path))[0]

        if output_txt_path is None:
            out_dir = os.path.join(self.base_data_dir, "output")
            os.makedirs(out_dir, exist_ok=True)
            output_txt_path = os.path.join(out_dir, f"{pdf_name}_summary.txt")
        else:
            os.makedirs(os.path.dirname(output_txt_path) or ".", exist_ok=True)

        sections = self.extract_sections(pdf_path, max_pages=max_pages)
        summary  = self.summarize_sections(sections)
        self._save(summary, output_txt_path)

        print("Summary agent completed successfully.")
        return output_txt_path


    #deals with duplicates headings, where a paragraph completeion is on the next page for examplee
    def _deduplicate_incremental_pages(self, doc, max_pages):
        MIN_NEW_CHARS = 30 #only accept the pae where the next page adds to it at least 30 characters.
        #for example: 
        #Page 1: Introduction
        #Page 2: Introduction. this should not be added
        n = len(doc) if max_pages is None else min(len(doc), max_pages)

        page_texts = []
        for i in range(n):
            raw = doc[i].get_text("text")
            normalised = re.sub(r"\s+", " ", raw).strip()
            # "/s+ means more than one white space, so any spaces will be replaced with one space
            page_texts.append(normalised)

        keep = []
        i = 0
        while i < n:
            while i + 1 < n and self._is_incremental_build(
                page_texts[i], page_texts[i + 1], MIN_NEW_CHARS
            ):
                i += 1  # page[i] is an animation step — skip, move to final
            keep.append(i)
            i += 1

        removed = n - len(keep)
        if removed:
            print(
                f"[SummaryAgent] Removed {removed} incremental-reveal duplicate "
                f"page(s). Kept {len(keep)} logical slide(s) out of {n} pages."
            )
        return keep

    #checks if the next page is an expanded version of the previous page
    def _is_incremental_build(self, text_a: str, text_b: str, min_new: int) -> bool:
        if not text_a:
            return False #if the page is empty, then not expanded
        if not text_b.startswith(text_a):  #if it doesnt begin with the beginning of page a, then not expanded
            return False
        if len(text_b) - len(text_a) < min_new: # if the next page adds less than 30 characters, skip it
            return False
        
        boundary_char = text_b[len(text_a)] #gets the first character after text A
        if boundary_char.isalnum(): #if it begins with a character or number,then it is not a boundary
            return False
        return True #anything else is considered an expanded boundary

#functoin used to extract sections, and detects headings, bodies, codes...
    def extract_sections(self, pdf_path, max_pages=None):
        doc       = fitz.open(pdf_path)
        num_pages = len(doc)
        n         = num_pages if max_pages is None else min(num_pages, max_pages)

        #removes duplicate texts and headings
        keep_indices = self._deduplicate_incremental_pages(doc, max_pages)
        
        #checks how many texts appears in a certain font size, returns a dictionary of  {size:count}
        size_counter  = Counter()
        
        #checks the style text as a whole, such as {(size,italic,bold): count} dictionary
        cluster_chars = defaultdict(int)
        all_spans     = []

        for page_num in keep_indices:
            page   = doc[page_num]
            blocks = page.get_text("dict")["blocks"]  #from myMyPDF, returns the pagetext as dictionary with font info
            #in the loop, for each block, break it down and and extract the font info the in the structure below
            for block in blocks: 
                for line in block.get("lines", []): 
                    for span in line.get("spans", []):
                        text = span["text"].strip()
                        if not text:
                            continue
                        size   = round(span["size"], 1)
                        flags  = span["flags"] #style information
                        italic = bool(flags & 2**1) #looks inside flags,returns true or false
                        bold   = bool(flags & 2**4)
                        size_counter[size]                   += len(text)
                        cluster_chars[(size, italic, bold)]  += len(text)
                        all_spans.append({
                            "text":   text,
                            "size":   size,
                            "font":   span["font"],
                            "italic": italic,
                            "bold":   bold,
                        })

        doc.close()

        if not size_counter: #if it cannot detect any sizes, then pdf is corrupted
            print("[SummaryAgent] No extractable text found. Is this a scanned PDF?")
            return [{"heading": "Content", "body": ""}]

        body_size = size_counter.most_common(1)[0][0] #takes the most common size as the body size
        print(
            f"[SummaryAgent] Detected body font size: {body_size}pt  "
            f"({len(keep_indices)} logical slide(s) after deduplication)"
        )
        #calls the classfication function to detect headings and bodies
        sections = self._classify_and_group(all_spans, body_size, cluster_chars)
        #if it cannot detect changes in font , then chunks using pages
        if not sections:
            print("[SummaryAgent] No sections found — falling back to page chunks.")
            raw = " ".join(s["text"] for s in all_spans)
            sections = self._chunk_raw_text(raw)

        print(f"[SummaryAgent] Extracted {len(sections)} section(s).")
        for i, sec in enumerate(sections, 1):
            preview = sec["body"][:60].replace("\n", " ")
            print(f"  {i}. '{sec['heading']}' — {len(sec['body'])} chars | {preview}")

        return sections

    def _classify_and_group(self, all_spans, body_size, cluster_chars):
        sections         = [] #final section classifcation
        current_heading  = "Introduction"
        current_body     = []
        pending_number   = "" #stores heading numbers
        prev_was_heading = False #stores wether a previous span was a number

        for span in all_spans:
            role = self._classify_span(#classifies each span as heading, noise, code, or body
                span, body_size, cluster_chars,
                has_pending_number=bool(pending_number),
            )
            text = span["text"]

            if role == "noise": #if it is a noise, ignore 
                continue

            if role == "heading": 
                if re.fullmatch(r"\d+(\.\d+)*\.?", text): #matches any digits, such as 2, 2.2, 2.2.2
                    pending_number = (pending_number + " " + text).strip()
                    prev_was_heading = False
                    continue

                full_title     = (pending_number + " " + text).strip()
                pending_number = ""

                if prev_was_heading and sections:
                    sections[-1]["heading"] += " " + full_title
                else:
                    body_text = self._assemble_body(current_body)
                  
                    if body_text and len(body_text) >= MIN_SECTION_CHARS:
                        sections.append({
                            "heading": current_heading,
                            "body":    body_text,
                        })
                    current_heading = full_title
                    current_body    = []

                prev_was_heading = True

            else:
                prev_was_heading = False
                pending_number   = ""
                current_body.append((role, text))

        # Flush the last section — apply the same threshold
        body_text = self._assemble_body(current_body)
        if body_text and len(body_text) >= MIN_SECTION_CHARS:
            sections.append({"heading": current_heading, "body": body_text})

        return sections

    def _classify_span(self, span, body_size, cluster_chars,
                       has_pending_number=False):
        text   = span["text"]
        size   = span["size"]
        italic = span["italic"]
        bold   = span["bold"]
        font   = span["font"].lower()

        cluster_key  = (size, italic, bold)
        total_chars  = cluster_chars[cluster_key]

        if any(m in font for m in MONO_FONTS):
            return "code"

        if re.fullmatch(r"\d{1,4}", text): #removes digits like 1 11 111 1111
            return "noise"

        if re.match( #ignores tables, figures, ect.. as they are exluded
            r"^(Listing|Figure|Fig\.|Table|Algorithm|Exhibit)\s+\d",
            text, re.IGNORECASE
        ):
            return "noise"

        if bold and italic and size <= body_size: #probably figure names, captions ect.. ignore
            return "noise"

        if size < body_size - 1.5: #anything less then boady size is ignored
            return "noise"

        if size >= body_size + 1.0 and total_chars < HEADING_CHAR_LIMIT:
            return "heading" #anything larger tham body size and less than max heading size is a heading

        if body_size + 0.3 <= size < body_size + 1.0 and total_chars < HEADING_CHAR_LIMIT:
            if re.match(r"^\d+[.\d]*\s", text): #if starts with number then text  like 1.1 intro , then heading
                return "heading"
            if re.fullmatch(r"\d+(\.\d+)*\.?", text): #if starts wih number only like 1.1
                return "heading"
            if has_pending_number: #if it has a pending number then heading
                return "heading"
            return "body"

        return "body"

    def _assemble_body(self, span_list):
        """example span list
        [
    ("body", "An agent perceives"),
    ("body", "its environment."),
    ("code", "print(action)")
]
        """
        if not span_list:
            return ""
        parts = []
        for role, text in span_list:
            parts.append(f"\n  {text}" if role == "code" else text)
        joined = " ".join(parts)
        joined = re.sub(r"(\w)-\s+(\w)", r"\1\2", joined) #fixes hyphenated words
        joined = re.sub(r"  +", " ", joined)
        return joined.strip()

    """ sample output:
    sections = [
    {
        "heading": "Agents",
        "body": "An agent perceives..."
    }
    ]
    """

    def summarize_sections(self, sections):
        all_summaries = []
        total = len(sections)

        for i, sec in enumerate(sections, 1):
            heading = sec["heading"]
            body    = sec["body"]
            print(f"[SummaryAgent] Summarising {i}/{total}: '{heading}'")

            chunks = self._chunk_text(body, MAX_CHUNK_CHARS)
            sub_summaries = []

            for j, chunk in enumerate(chunks):
                if len(chunks) > 1:
                    print(f"  chunk {j+1}/{len(chunks)} ({len(chunk)} chars)")
                prompt = f"Section heading: {heading}\n\n{chunk}"
                sub_summaries.append(self._call_llm(prompt))

            all_summaries.append(f"## {heading}\n\n" + "\n\n".join(sub_summaries))

        return "\n\n---\n\n".join(all_summaries)

    def _call_llm(self, text):
        response = ollama.chat(
            model=self.model_name,
            messages=[
                {"role": "system", "content": SUMMARY_PROMPT}, #system role tells AI how to behave
                {
                    "role": "user",
                    "content": ( 
                        "Rewrite and condense this academic text while keeping about 50-60% "
                        "of the information. Do not over-shorten it. Keep the important "
                        "definitions, explanations, and examples.\n\n"
                        f"{text}"
                    ),
                },
            ],
        )
        #content contains the actual text. ollama returns dictionary
        return response["message"]["content"].strip()

    def _chunk_text(self, text, max_chars):
        if len(text) <= max_chars:
            return [text]
        paragraphs = re.split(r"\n{2,}", text) #splits text by blank lines
        chunks, current, current_len = [], [], 0
        for para in paragraphs:
            para = para.strip()
            if not para:
                continue
            if current_len + len(para) > max_chars and current:
                chunks.append("\n\n".join(current))
                current, current_len = [], 0
            current.append(para)
            current_len += len(para)
        if current:
            chunks.append("\n\n".join(current))
        return chunks if chunks else [text]

    #if chunking by detecting headings doesnt work, chunks it deterministically
    def _chunk_raw_text(self, raw_text):
        chunks = self._chunk_text(raw_text, MAX_CHUNK_CHARS)
        return [{"heading": f"Part {i+1}", "body": c} for i, c in enumerate(chunks)]

    def _save(self, text, path): #creates a folder and writes in it
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            f.write(text)
        print(f"[SummaryAgent] Summary saved to: {path}")

if __name__ == "__main__":
    agent = SummaryAgent(model_name="qwen2.5:7b")
    result_path = agent.run(
        pdf_path=r"Data/input/textbook/chapter1.pdf",
        output_txt_path=r"Data/output/chapter1_summary.txt",
    )
    print(f"\nFinal summary file: {result_path}")
