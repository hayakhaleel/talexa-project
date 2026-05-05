import os
import subprocess
import ollama
import pdfplumber
import sys
import re
from pathlib import Path

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from Prompts.slides_latex_prompt import SLIDES_LATEX_PROMPT


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class LatexAgent:

    def __init__(self, model_name="qwen2.5:7b"):
        self.model_name = model_name

    def run(self, input_pdf_path, output_filename=None, output_dir=None, max_attempts=2):
        if output_dir is None:
            output_dir = str(PROJECT_ROOT / "Data" / "intermediate")
        os.makedirs(output_dir, exist_ok=True)

        tex_path = os.path.join(output_dir, f"{output_filename}.tex")

        print(f"Extracting text from slide PDF: {input_pdf_path}")

        chunks, total_slides = self._extract_and_chunk_pdf(input_pdf_path, chunk_size=2)
        print(f"Total slides found: {total_slides}. Broken into {len(chunks)} chunks for safe processing.")

        attempt = 0
        while attempt < max_attempts:
            print(f"\nAttempt {attempt + 1} of {max_attempts}")
            all_latex_frames = []

            for i, chunk in enumerate(chunks):
                slide_count = chunk.count("--- SLIDE")
                print(f"Generating Beamer frames for batch {i+1}/{len(chunks)} (up to {slide_count} slides) via {self.model_name}...")

                try:
                    response = ollama.chat(
                        model=self.model_name,
                        messages=[
                            {
                                "role": "system",
                                "content": SLIDES_LATEX_PROMPT
                            },
                            {"role": "user", "content": f"SLIDE TEXT CHUNK:\n{chunk}"}
                        ],
                        options={"num_predict": 8192}
                    )

                    raw_output = response["message"]["content"].strip()
                    clean_frames = self._clean_frames_output(raw_output)
                    all_latex_frames.append(clean_frames)

                except Exception as e:
                    print(f"Error generating batch {i+1}: {e}. Skipping this chunk.")

            print("Assembling all chunks into final LaTeX document...")
            latex_code = self._assemble_full_document(all_latex_frames)

            with open(tex_path, "w", encoding="utf-8") as f:
                f.write(latex_code)

            pdf_path = self._compile_to_pdf(tex_path, output_dir, output_filename)

            if pdf_path:
                print("Agent completed successfully.")

                # cleanup intermediate directory (keep only .tex and .pdf)
                for file in os.listdir(output_dir):
                    file_path = os.path.join(output_dir, file)
                    if os.path.isfile(file_path):
                        ext = os.path.splitext(file)[1].lower()
                        if ext not in [".tex", ".pdf"]:
                            try:
                                os.remove(file_path)
                            except Exception as e:
                                print(f"Could not delete {file}: {e}")

                return pdf_path

            attempt += 1
            if attempt < max_attempts:
                print("Retrying LaTeX generation due to compilation failure...\n")

        print("Agent failed after multiple attempts.")
        return None

    def _extract_and_chunk_pdf(self, pdf_path, chunk_size=4):
        chunks = []
        current_chunk = []
        total_pages = 0

        with pdfplumber.open(pdf_path) as pdf:
            total_pages = len(pdf.pages)

            for i, page in enumerate(pdf.pages):
                text = page.extract_text()
                if not text:
                    continue

                lines = [line.strip() for line in text.splitlines() if line.strip()]
                if not lines:
                    continue

                cleaned_lines = []

                for line in lines:
                    lower = line.lower()

                    if "prepared by" in lower:
                        continue
                    if "@" in line:
                        continue
                    if line.isdigit():
                        continue
                    if re.fullmatch(r".*\b\d+\b", line) and len(line.split()) <= 4:
                        continue
                    if "copyright" in lower or line.startswith("©"):
                        continue

                    cleaned_lines.append(line)

                if not cleaned_lines:
                    continue

                slide_title = cleaned_lines[0]
                slide_body = cleaned_lines[1:]

                if not slide_body:
                    continue

                slide_text = (
                    f"--- SLIDE {i+1} ---\n"
                    f"TITLE: {slide_title}\n"
                    f"BODY:\n" + "\n".join(slide_body)
                )

                current_chunk.append(slide_text)

                if len(current_chunk) == chunk_size:
                    chunks.append("\n\n".join(current_chunk))
                    current_chunk = []

        if current_chunk:
            chunks.append("\n\n".join(current_chunk))

        return chunks, total_pages

    def _clean_frames_output(self, text):
        backticks = "`" * 3
        text = re.sub(rf"{backticks}(?:latex)?", "", text)
        text = text.replace(backticks, "")
        text = text.strip()

        begin_count = text.count(r"\begin{frame}")
        end_count = text.count(r"\end{frame}")

        if begin_count > end_count:
            missing = begin_count - end_count
            text += "\n\\end{frame}" * missing

        return text

    def _assemble_full_document(self, frames_list):
        combined_frames = "\n\n".join(frames_list)

        title_match = re.search(r'\\begin\{frame\}\{([^}]*)\}', combined_frames)
        document_title = title_match.group(1).strip() if title_match else "Lecture Slides"

        document = (
            rf"""\documentclass{{beamer}}

\title{{{document_title}}}

\begin{{document}}

\begin{{frame}}
    \titlepage
\end{{frame}}

"""
            + combined_frames
            + r"""

\end{document}
"""
        )

        return document

    def _compile_to_pdf(self, tex_path, output_dir, output_name):
        print(f"Compiling {output_name}.pdf...")
        pdf_path = os.path.join(output_dir, f"{output_name}.pdf")

        try:
            for _ in range(2):
                result = subprocess.run(
                    ["pdflatex", "-interaction=nonstopmode", f"-output-directory={output_dir}", tex_path],
                    capture_output=True,
                    text=True,
                    encoding="utf-8",
                    errors="replace"
                )

            if os.path.exists(pdf_path):
                print(f"Success! View result: {pdf_path}")
                if result.returncode != 0:
                    print("Note: LaTeX finished with minor warnings, but the PDF was generated successfully.")
                return pdf_path
            else:
                print(f"PDF Compilation completely failed for {output_name}.")
                print("--- LaTeX Error Tail ---")
                if result.stdout:
                    print("\n".join(result.stdout.splitlines()[-15:]))
                return None

        except Exception as e:
            print(f"Execution Error: {e}")
            return None


if __name__ == "__main__":
    input_file = str(PROJECT_ROOT / "Data" / "input" / "slides" / "Introduction to AI.pdf")

    if os.path.exists(input_file):

        agent = LatexAgent()

        output_name = os.path.splitext(os.path.basename(input_file))[0]

        agent.run(input_file, output_name)

    else:
        print(f"Input file not found at: {input_file}")
