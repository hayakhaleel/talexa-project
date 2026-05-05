import os
import subprocess
import sys
from pathlib import Path

import ollama

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _load_latex_prompt():
    prompt_file = Path(__file__).resolve().parents[1] / "Prompts" / "latex_prompt.py"
    namespace = {}
    exec(prompt_file.read_text(encoding="utf-8"), namespace)
    return namespace["LATEX_PROMPT"]


LATEX_PROMPT = _load_latex_prompt()


class LatexAgent:

    def __init__(self, model="qwen2.5:7b"):
        self.model = model


    def escape_latex(self, text):

        replacements = {
            "&": r"\&",
            "%": r"\%",
            "$": r"\$",
            "#": r"\#",
            "_": r"\_",
        }

        for k, v in replacements.items():
            text = text.replace(k, v)

        return text


    def generate_latex(self, summary_path):

        if not os.path.exists(summary_path):
            raise FileNotFoundError(f"Summary file not found: {summary_path}")

        with open(summary_path, "r", encoding="utf-8") as f:
            summary_text = f.read()

        summary_text = self.escape_latex(summary_text)

        response = ollama.chat(
            model=self.model,
            messages=[
                {"role": "system", "content": LATEX_PROMPT},
                {"role": "user", "content": f"summary_text:\n{summary_text}"}
            ]
        )

        latex_code = response["message"]["content"]
        latex_code = latex_code.replace("```latex", "").replace("```", "")

        return latex_code.strip()


    def save_latex(self, latex_code, output_path):

        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        with open(output_path, "w", encoding="utf-8") as f:
            f.write(latex_code)

        print(f"LaTeX file saved at: {output_path}")


    def compile_pdf(self, tex_path):

        folder = os.path.dirname(tex_path)
        tex_file = os.path.basename(tex_path)

        result = subprocess.run(
            ["pdflatex", "-interaction=nonstopmode", tex_file],
            cwd=folder,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )

        if result.returncode == 0:
            print("PDF compiled successfully.")
            return True
        else:
            print("LaTeX compilation failed.")
            return False


    def run(
        self,
        pdf_name=None,
        summary_path=None,
        output_tex_path=None,
        max_attempts=3,
        compile_pdf=False,
    ):
        if summary_path is None:
            if not pdf_name:
                raise ValueError("Either pdf_name or summary_path must be provided.")
            summary_path = f"Data/output/{pdf_name}_summary.txt"

        if output_tex_path is None:
            if pdf_name:
                output_tex_path = f"Data/output/{pdf_name}.tex"
            else:
                summary_stem = Path(summary_path).stem
                cleaned_stem = summary_stem.removeprefix("Summary_")
                output_tex_path = f"Data/output/Latex_{cleaned_stem}.tex"

        attempt = 0

        while attempt < max_attempts:

            print(f"\nAttempt {attempt + 1} of {max_attempts}")

            latex_code = self.generate_latex(summary_path)

            self.save_latex(latex_code, output_tex_path)

            if not compile_pdf:
                print("Agent completed successfully.")
                return output_tex_path

            success = self.compile_pdf(output_tex_path)

            if success:
                print("Agent completed successfully.")
                return output_tex_path

            attempt += 1
            print("Retrying LaTeX generation...\n")

        print("Agent failed after multiple attempts.")
        return None

if __name__ == "__main__":
    agent = LatexAgent()
    agent.run("lecture1")
