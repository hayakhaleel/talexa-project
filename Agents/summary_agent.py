# Agents/summary_agent.py

import os
import re
import fitz
import ollama

from Prompts.summary_prompt import SUMMARY_PROMPT


class SummaryAgent:

    def __init__(self, model_name="qwen2.5:7b", base_data_dir="Data"):
        self.model_name = model_name
        self.base_data_dir = base_data_dir

    def extract_pdf_text(self, pdf_path, max_pages=None):
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"PDF not found: {pdf_path}")

        doc = fitz.open(pdf_path)

        total_pages = len(doc)
        n = total_pages if max_pages is None else min(total_pages, max_pages)

        full_text = ""

        for i in range(n):
            page_text = doc[i].get_text("text")
            full_text += f"\n\n===== PAGE {i+1} =====\n\n"
            if page_text:
                full_text += page_text

        doc.close()
        print("Text extracted successfully.")

        return full_text


    def _is_likely_heading(self, line):
        stripped = line.strip()
        if not stripped:
            return False

        if stripped.startswith("===== PAGE"):
            return False

        if re.fullmatch(r"\d+", stripped):
            return False

        # Numbered headings such as "1 Introduction" or "2.3 Security Models"
        if re.fullmatch(r"\d+(?:\.\d+)*\.?\s+[A-Za-z].*", stripped):
            return True

        words = stripped.split()
        word_count = len(words)
        if word_count > 8:
            return False

        # Chapter titles like "CHAPTER 10"
        if stripped.isupper() and word_count <= 4:
            return True

        # Short title-case headings like "People", "Processes", "Always Be Paranoid"
        alpha_words = [word for word in words if re.search(r"[A-Za-z]", word)]
        if not alpha_words:
            return False

        title_case_ratio = sum(
            1 for word in alpha_words if word[:1].isupper()
        ) / len(alpha_words)

        return title_case_ratio >= 0.8

    def split_by_sections(self, text):
        lines = text.splitlines()
        heading_indices = []

        for idx, line in enumerate(lines):
            if not self._is_likely_heading(line):
                continue

            prev_line = lines[idx - 1].strip() if idx > 0 else ""
            next_line = lines[idx + 1].strip() if idx + 1 < len(lines) else ""

            # A heading should usually be visually separated and followed by content.
            if next_line.startswith("===== PAGE") or next_line == "":
                continue

            if prev_line and not prev_line.startswith("===== PAGE") and len(prev_line.split()) > 12:
                continue

            heading_indices.append(idx)

        print("\nDetected section headings:")
        for i, idx in enumerate(heading_indices, 1):
            print(f"{i}. {lines[idx].strip()}")

        if not heading_indices:
            return [text.strip()]

        sections = []

        first_start = heading_indices[0]
        intro = "\n".join(lines[:first_start]).strip()
        if intro:
            sections.append(intro)

        for i, start_idx in enumerate(heading_indices):
            end_idx = heading_indices[i + 1] if i + 1 < len(heading_indices) else len(lines)
            section_text = "\n".join(lines[start_idx:end_idx]).strip()
            if section_text:
                sections.append(section_text)

        return sections


    def generate_summary(self, doc_text):
        response = ollama.chat(
            model=self.model_name,
            messages=[
                {"role": "system", "content": SUMMARY_PROMPT},
                {
                    "role": "user",
                    "content": (
                        "Rewrite and condense this academic text while keeping about 50–60% "
                        "of the information. Do not over-shorten it. Keep the important "
                        "definitions, explanations, and examples.\n\n"
                        f"{doc_text}"
                    )
                },
            ],
        )
        summary_text = response["message"]["content"].strip()
        return summary_text


    def summarize_sections(self, doc_text):
        sections = self.split_by_sections(doc_text)
        summaries = []

        print(f"Detected {len(sections)} section blocks.")

        for i, section in enumerate(sections, 1):
            print(f"Summarizing section {i}/{len(sections)}...")
            section_summary = self.generate_summary(section)
            summaries.append(section_summary)

        return "\n\n".join(summaries)


    def save_summary(self, summary_text, output_txt_path):
        os.makedirs(os.path.dirname(output_txt_path) or ".", exist_ok=True)

        with open(output_txt_path, "w", encoding="utf-8") as f:
            f.write(summary_text)

        print(f"Summary saved to: {output_txt_path}")


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

        doc_text = self.extract_pdf_text(pdf_path, max_pages=max_pages)
        summary_text = self.summarize_sections(doc_text)
        self.save_summary(summary_text, output_txt_path)

        print("Summary agent completed successfully.")
        return output_txt_path


if __name__ == "__main__":
    agent = SummaryAgent(model_name="qwen2.5:7b")

    result_path = agent.run(
        pdf_path=r"Data/input/slides/AI_ch1.pdf",
        output_txt_path=r"Data/output/AI_ch2_summary.txt",
        max_pages=None
    )

    print(f"\nFinal summary file: {result_path}")
