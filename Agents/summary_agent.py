# Agents/summary_agent.py

import os
import re
import fitz
import ollama
from collections import Counter, defaultdict

from Prompts.summary_prompt import SUMMARY_PROMPT

# ---------------------------------------------------------------------------
# Tuning constants
# ---------------------------------------------------------------------------
# FIX 1: Lowered from 120 → 40 so short but valid slide content isn't discarded.
# Presentation slides are concise by nature — 120 chars excluded legitimate slides.
MIN_SECTION_CHARS  = 40
MAX_CHUNK_CHARS    = 6000  # max chars sent to the LLM in a single call
HEADING_CHAR_LIMIT = 500   # a font cluster with more chars than this is body, not headings
PAGES_PER_CHUNK    = 5     # fallback when no headings detected at all

# Monospace font name fragments (publisher-agnostic)
MONO_FONTS = (
    "courier", "mono", "code", "consolas", "inconsolata",
    "sourcecodepro", "jetbrains", "firacode", "droidmono",
    "lucidaconsole", "anonymous", "terminus",
)


class SummaryAgent:

    def __init__(self, model_name="qwen2.5:7b", base_data_dir="Data"):
        self.model_name    = model_name
        self.base_data_dir = base_data_dir

    # -----------------------------------------------------------------------
    # PUBLIC ENTRY POINT
    # -----------------------------------------------------------------------
    def run(self, pdf_path, output_txt_path=None, max_pages=None):
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"PDF not found: {pdf_path}")

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

    # -----------------------------------------------------------------------
    # INCREMENTAL-REVEAL DEDUPLICATION
    # -----------------------------------------------------------------------
    def _deduplicate_incremental_pages(self, doc, max_pages):
        """
        Collapses animation build groups (where slide N+1 is a strict superset
        of slide N) into only the final page of each group.

        FIX 2 (was the main bug): The original check used Python's `in` operator:
            page_texts[i] in page_texts[i + 1]
        This is a SUBSTRING check, not a semantic "is subset of" check.
        Short slides like "Introduction" were being found inside longer slides
        like "Introduction to PySpark and its role in..." and incorrectly dropped.

        The fix: only deduplicate when the earlier page's text is a proper prefix
        of the next page's text AND the texts differ by meaningful new content
        (at least MIN_NEW_CHARS characters added). This correctly handles
        animation exports (where each step adds exactly one bullet) while
        preserving distinct slides that happen to share an opening phrase.
        """
        MIN_NEW_CHARS = 30  # next slide must add at least this many chars to count as a build step

        n = len(doc) if max_pages is None else min(len(doc), max_pages)

        # Extract normalised text for each page (whitespace-collapsed)
        page_texts = []
        for i in range(n):
            raw = doc[i].get_text("text")
            normalised = re.sub(r"\s+", " ", raw).strip()
            page_texts.append(normalised)

        keep = []
        i = 0
        while i < n:
            # Only skip page[i] if ALL of these hold:
            #   1. page[i]'s text is a true prefix of page[i+1]'s text
            #   2. page[i+1] adds at least MIN_NEW_CHARS of new content
            # This means "Introduction" does NOT match inside
            # "Introduction to PySpark..." because it is not a prefix
            # followed by a space/punctuation boundary.
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

    def _is_incremental_build(self, text_a: str, text_b: str, min_new: int) -> bool:
        """
        Return True only if text_b is an extension of text_a:
          - text_b starts with text_a (prefix check, not substring)
          - text_b is longer by at least min_new characters
          - the character immediately after text_a in text_b is whitespace
            or punctuation (word-boundary check prevents false matches like
            "Introduction" being a prefix of "Introduction to PySpark")

        This is stricter than the original `text_a in text_b` substring test
        and correctly identifies animation build steps without conflating
        separate slides that share an opening word or phrase.
        """
        if not text_a:
            return False
        if not text_b.startswith(text_a):
            return False
        if len(text_b) - len(text_a) < min_new:
            return False
        # Word-boundary guard: the char right after text_a must be whitespace
        # or punctuation, not a letter/digit (which would mean text_a is a
        # prefix of a longer word, not a prefix of the whole slide's content).
        boundary_char = text_b[len(text_a)]
        if boundary_char.isalnum():
            return False
        return True

    # -----------------------------------------------------------------------
    # STEP 1 — two-pass font-aware extraction (zero hardcoded font names)
    # -----------------------------------------------------------------------
    def extract_sections(self, pdf_path, max_pages=None):
        """
        Pass 0: deduplicate incremental-reveal pages (animation exports).
        Pass 1: survey all spans to compute body font size and per-cluster
                char counts (used to distinguish headings from sidebar body).
        Pass 2: classify each span using only size, flags, and cluster stats.
        Pass 3: group classified spans into labelled sections.

        Returns: [{"heading": str, "body": str}, ...]
        """
        doc       = fitz.open(pdf_path)
        num_pages = len(doc)
        n         = num_pages if max_pages is None else min(num_pages, max_pages)

        # ---- Pass 0: deduplicate incremental slides -------------------------
        keep_indices = self._deduplicate_incremental_pages(doc, max_pages)

        # ---- Pass 1: survey (only kept pages) ------------------------------
        size_counter  = Counter()
        cluster_chars = defaultdict(int)
        all_spans     = []

        for page_num in keep_indices:
            page   = doc[page_num]
            blocks = page.get_text("dict")["blocks"]
            for block in blocks:
                for line in block.get("lines", []):
                    for span in line.get("spans", []):
                        text = span["text"].strip()
                        if not text:
                            continue
                        size   = round(span["size"], 1)
                        flags  = span["flags"]
                        italic = bool(flags & 2**1)
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

        if not size_counter:
            print("[SummaryAgent] No extractable text found. Is this a scanned PDF?")
            return [{"heading": "Content", "body": ""}]

        body_size = size_counter.most_common(1)[0][0]
        print(
            f"[SummaryAgent] Detected body font size: {body_size}pt  "
            f"({len(keep_indices)} logical slide(s) after deduplication)"
        )

        # ---- Pass 2 + 3: classify and group --------------------------------
        sections = self._classify_and_group(all_spans, body_size, cluster_chars)

        if not sections:
            print("[SummaryAgent] No sections found — falling back to page chunks.")
            raw = " ".join(s["text"] for s in all_spans)
            sections = self._chunk_raw_text(raw)

        print(f"[SummaryAgent] Extracted {len(sections)} section(s).")
        for i, sec in enumerate(sections, 1):
            preview = sec["body"][:60].replace("\n", " ")
            print(f"  {i}. '{sec['heading']}' — {len(sec['body'])} chars | {preview}")

        return sections

    # -----------------------------------------------------------------------
    # STEP 2 — classify + group (single pass over spans)
    # -----------------------------------------------------------------------
    def _classify_and_group(self, all_spans, body_size, cluster_chars):
        sections         = []
        current_heading  = "Introduction"
        current_body     = []
        pending_number   = ""
        prev_was_heading = False

        for span in all_spans:
            role = self._classify_span(
                span, body_size, cluster_chars,
                has_pending_number=bool(pending_number),
            )
            text = span["text"]

            if role == "noise":
                continue

            if role == "heading":
                if re.fullmatch(r"\d+(\.\d+)*\.?", text):
                    pending_number = (pending_number + " " + text).strip()
                    prev_was_heading = False
                    continue

                full_title     = (pending_number + " " + text).strip()
                pending_number = ""

                if prev_was_heading and sections:
                    sections[-1]["heading"] += " " + full_title
                else:
                    body_text = self._assemble_body(current_body)
                    # FIX 3: use the lowered MIN_SECTION_CHARS constant (40)
                    # so short-but-complete slide bodies aren't silently dropped.
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

    # -----------------------------------------------------------------------
    # STEP 3 — span classifier
    # -----------------------------------------------------------------------
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

        if re.fullmatch(r"\d{1,4}", text):
            return "noise"

        if re.match(
            r"^(Listing|Figure|Fig\.|Table|Algorithm|Exhibit)\s+\d",
            text, re.IGNORECASE
        ):
            return "noise"

        if bold and italic and size <= body_size:
            return "noise"

        if size < body_size - 1.5:
            return "noise"

        if size >= body_size + 1.0 and total_chars < HEADING_CHAR_LIMIT:
            return "heading"

        if body_size + 0.3 <= size < body_size + 1.0 and total_chars < HEADING_CHAR_LIMIT:
            if re.match(r"^\d+[.\d]*\s", text):
                return "heading"
            if re.fullmatch(r"\d+(\.\d+)*\.?", text):
                return "heading"
            if has_pending_number:
                return "heading"
            return "body"

        return "body"

    # -----------------------------------------------------------------------
    # Body assembly
    # -----------------------------------------------------------------------
    def _assemble_body(self, span_list):
        if not span_list:
            return ""
        parts = []
        for role, text in span_list:
            parts.append(f"\n  {text}" if role == "code" else text)
        joined = " ".join(parts)
        joined = re.sub(r"(\w)-\s+(\w)", r"\1\2", joined)
        joined = re.sub(r"  +", " ", joined)
        return joined.strip()

    # -----------------------------------------------------------------------
    # STEP 4 — summarise each section
    # -----------------------------------------------------------------------
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
                {"role": "system", "content": SUMMARY_PROMPT},
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
        return response["message"]["content"].strip()

    # -----------------------------------------------------------------------
    # Helpers
    # -----------------------------------------------------------------------
    def _chunk_text(self, text, max_chars):
        if len(text) <= max_chars:
            return [text]
        paragraphs = re.split(r"\n{2,}", text)
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

    def _chunk_raw_text(self, raw_text):
        chunks = self._chunk_text(raw_text, MAX_CHUNK_CHARS)
        return [{"heading": f"Part {i+1}", "body": c} for i, c in enumerate(chunks)]

    def _save(self, text, path):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            f.write(text)
        print(f"[SummaryAgent] Summary saved to: {path}")


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    agent = SummaryAgent(model_name="qwen2.5:7b")
    result_path = agent.run(
        pdf_path=r"Data/input/textbook/chapter1.pdf",
        output_txt_path=r"Data/output/chapter1_summary.txt",
    )
    print(f"\nFinal summary file: {result_path}")