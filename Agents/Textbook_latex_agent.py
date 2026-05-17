import os
import re
import subprocess
import sys
from pathlib import Path

import ollama

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

#loads latex proompt
def _load_latex_prompt():
    prompt_file = Path(__file__).resolve().parents[1] / "Prompts" / "latex_prompt.py"
    namespace = {}
    exec(prompt_file.read_text(encoding="utf-8"), namespace)
    return namespace["LATEX_PROMPT"]


LATEX_PROMPT = _load_latex_prompt()

#each model call is 4000 tokens 
CHUNK_SIZE = 4000  # max chars per LLM call


class LatexAgent:

    def __init__(self, model="qwen2.5:7b"):
        self.model = model


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

        for attempt in range(max_attempts):
            print(f"\nAttempt {attempt + 1} of {max_attempts}")

            latex_code = self.generate_latex(summary_path)
            self.save_latex(latex_code, output_tex_path)

            if not compile_pdf:
                print("Agent completed successfully.")
                return output_tex_path

            if self.compile_pdf(output_tex_path):
                print("Agent completed successfully.")
                return output_tex_path

            print("Retrying LaTeX generation...\n")

        print("Agent failed after multiple attempts.")
        return None

    def generate_latex(self, summary_path):
        if not os.path.exists(summary_path):
            raise FileNotFoundError(f"Summary file not found: {summary_path}")

        with open(summary_path, "r", encoding="utf-8") as f:
            summary_text = f.read()

        summary_text = self.escape_latex(summary_text)
        chunks = self._chunk_summary(summary_text)
        print(f"[LatexAgent] Summary split into {len(chunks)} chunk(s).")

        all_frames = []
        for i, chunk in enumerate(chunks, 1):
            print(f"[LatexAgent] Generating frames for chunk {i}/{len(chunks)}...")
            response = ollama.chat(
                model=self.model,
                messages=[
                    {"role": "system", "content": LATEX_PROMPT},
                    {"role": "user", "content": f"SUMMARY TEXT:\n{chunk}"},
                ],
                options={"num_predict": 8192},
            )
            raw = response["message"]["content"]
            frames = self._clean_frames(raw)
            all_frames.append(frames)

        return self._assemble_document(all_frames)

    
    def _clean_frames(self, text):
    

        # ── Strip markdown fences
        text = re.sub(r'```(?:latex)?', '', text, flags=re.IGNORECASE)
        text = text.replace('```', '').strip()

        # ── 1. Markdown bold to LaTeX bold
        text = re.sub(r'\*\*(.+?)\*\*', r'\\textbf{\1}', text)

        # ── 2. Missing backslash on \item 
        text = re.sub(r'(?m)^(\s*)(?<!\\)item\b', r'\1\\item', text)

        def replace_verbatim(m):
            content = m.group(1).strip()
            content = content.replace('\\', '\\textbackslash{}')
            content = content.replace('{', '\\{').replace('}', '\\}')
            content = content.replace('_', '\\_').replace('%', '\\%')
            content = content.replace('&', '\\&').replace('#', '\\#')
            content = content.replace('$', '\\$').replace('^', '\\^{}')
            lines = [
                f'\\texttt{{{line.strip()}}}'
                for line in content.splitlines()
                if line.strip()
            ]
            return '\n'.join(lines)

        text = re.sub(
            r'\\begin\{verbatim\}(.*?)\\end\{verbatim\}',
            replace_verbatim,
            text,
            flags=re.DOTALL,
        )

        # ── 4. Markdown dash bullets → \item ───────────────────────────────
        text = re.sub(r'(?m)^(\s+)- (.+)$', r'\1\\item \2', text)

        # ── 5. Unescaped & outside LaTeX commands ──────────────────────────
        lines = text.splitlines()
        fixed = []
        for line in lines:
            stripped = line.strip()
            if not stripped or stripped.startswith('\\') or stripped.startswith('%'):
                fixed.append(line)
                continue
            line = re.sub(r'(?<!\\)&', r'\\&', line)
            fixed.append(line)
        text = '\n'.join(fixed)

        # ── 6. Fix unbalanced frame tags
        begin_count = text.count(r'\begin{frame}')
        end_count   = text.count(r'\end{frame}')
        if begin_count > end_count:
            text += '\n\\end{frame}' * (begin_count - end_count)

        return text.strip()

   
    def _assemble_document(self, frames_list):
        combined = '\n\n'.join(frames_list)

        # Extract a title from the first frame
        title_match = re.search(r'\\begin\{frame\}\{([^}]+)\}', combined)
        title = title_match.group(1).strip() if title_match else 'Lecture Slides'

        return (
            rf"""\documentclass[t]{{beamer}}

\title{{{title}}}
\author{{Talexa}}
\date{{}}

\begin{{document}}

\begin{{frame}}[plain,noframenumbering]
    \titlepage
\end{{frame}}

"""
            + combined
            + '\n\n\\end{document}\n'
        )

 
    def escape_latex(self, text):
        """Escape characters that are dangerous in LaTeX prose but not in commands."""
        replacements = {
            '&': r'\&',
            '%': r'\%',
            '$': r'\$',
            '#': r'\#',
            '_': r'\_',
        }
        for k, v in replacements.items():
            text = text.replace(k, v)
        return text

    def _chunk_summary(self, text, chunk_size=CHUNK_SIZE):
        """Split on paragraph boundaries, never exceeding chunk_size chars."""
        if len(text) <= chunk_size:
            return [text]

        paragraphs = re.split(r'\n{2,}', text)
        chunks, current, current_len = [], [], 0

        for para in paragraphs:
            para = para.strip()
            if not para:
                continue
            if current_len + len(para) > chunk_size and current:
                chunks.append('\n\n'.join(current))
                current, current_len = [], 0
            current.append(para)
            current_len += len(para)

        if current:
            chunks.append('\n\n'.join(current))

        return chunks if chunks else [text]

    def save_latex(self, latex_code, output_path):
        os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(latex_code)
        print(f'LaTeX file saved at: {output_path}')

    def compile_pdf(self, tex_path):
        folder   = os.path.dirname(tex_path)
        tex_file = os.path.basename(tex_path)
        result   = subprocess.run(
            ['pdflatex', '-interaction=nonstopmode', tex_file],
            cwd=folder,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        if result.returncode == 0:
            print('PDF compiled successfully.')
            return True
        print('LaTeX compilation failed.')
        return False


if __name__ == '__main__':
    agent = LatexAgent()
    agent.run('lecture1')
