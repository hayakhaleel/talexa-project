# Agents/Slider_Builder_Agent.py

"""
Slide Beamer Code Generation Agent
Uses Ollama and runs on the output of the LatexAgent.
"""

import os
import re
import fitz
import json
import string
import bisect
import subprocess
import shutil
import httpx

from os import path
from pathlib import Path
from bisect import bisect_right
from typing import Sequence, Tuple, Optional

import ollama
from PIL import Image, ImageDraw, ImageFont


class SlideBuilderAgent:

    def __init__(
        self,
        llm_model="qwen2.5:7b",
        vlm_model="qwen2.5vl:7b",
        slide_prompt_path=str(Path(__file__).resolve().parents[1] / "Prompts" / "Slider_Builder_Prompt.py"),
        correct_prompt_path=str(Path(__file__).resolve().parents[1] / "Prompts" / "slide_beamer_correct.py"),
        select_proposal_prompt_path=str(Path(__file__).resolve().parents[1] / "Prompts" / "select_proposal.py"),
        ollama_host=None,
        ollama_timeout=None,
    ):
        self.llm_model = llm_model
        self.vlm_model = vlm_model
        self.slide_prompt_path = slide_prompt_path
        self.correct_prompt_path = correct_prompt_path
        self.select_proposal_prompt_path = select_proposal_prompt_path
        timeout_seconds = ollama_timeout
        if timeout_seconds is None:
            timeout_seconds = float(os.getenv("OLLAMA_TIMEOUT", "600"))
        self.ollama_timeout = timeout_seconds
        self.ollama_client = ollama.Client(
            host=ollama_host,
            timeout=httpx.Timeout(timeout_seconds),
        )

    def extract_json_block(self, text: str, first_only: bool = True):
        pattern = r"```json\s*([\s\S]*?)\s*```"
        matches = re.findall(pattern, text, flags=re.IGNORECASE)
        if first_only:
            return matches[0] if matches else text
        return matches

    def extract_beamer_code(self, tex_str):
        match = re.search(
            r"(\\documentclass(?:\[[^\]]*\])?\{beamer\}.*?\\end\{document\})",
            tex_str,
            re.DOTALL,
        )
        return match.group(1) if match else None

    def academic_design_block(self):
        return r"""
% Talexa academic visual design
\usetheme{Madrid}
\useinnertheme{rounded}
\usefonttheme{professionalfonts}
\setbeamertemplate{navigation symbols}{}

\definecolor{TalexaNavy}{HTML}{17324D}
\definecolor{TalexaTeal}{HTML}{1F6F78}
\definecolor{TalexaGold}{HTML}{C59B45}
\definecolor{TalexaMist}{HTML}{F4F7F8}
\definecolor{TalexaSlate}{HTML}{243746}

\setbeamercolor{background canvas}{bg=white}
\setbeamercolor{normal text}{fg=TalexaSlate,bg=white}
\setbeamercolor{structure}{fg=TalexaNavy}
\setbeamercolor{title}{fg=white,bg=TalexaNavy}
\setbeamercolor{frametitle}{fg=white,bg=TalexaNavy}
\setbeamercolor{palette primary}{fg=white,bg=TalexaNavy}
\setbeamercolor{palette secondary}{fg=white,bg=TalexaTeal}
\setbeamercolor{palette tertiary}{fg=white,bg=TalexaSlate}
\setbeamercolor{block title}{fg=white,bg=TalexaTeal}
\setbeamercolor{block body}{fg=TalexaSlate,bg=TalexaMist}
\setbeamercolor{alerted text}{fg=TalexaGold}
\setbeamercolor{item projected}{fg=white,bg=TalexaTeal}

\setbeamerfont{title}{series=\bfseries,size=\LARGE}
\setbeamerfont{author}{size=\normalsize}
\setbeamerfont{date}{size=\small}
\setbeamerfont{frametitle}{series=\bfseries,size=\large}
\setbeamerfont{block title}{series=\bfseries}

\setbeamertemplate{items}[circle]
\setbeamertemplate{blocks}[rounded][shadow=false]

\setbeamertemplate{frametitle}{
  \nointerlineskip
  \begin{beamercolorbox}[wd=\paperwidth,ht=3.2ex,dp=1.2ex,leftskip=1em,rightskip=1em]{frametitle}
    \usebeamerfont{frametitle}\insertframetitle
  \end{beamercolorbox}
  \vspace{0.2em}
  \hspace*{1em}{\color{TalexaGold}\rule{0.22\paperwidth}{0.8pt}}
}

\setbeamertemplate{footline}{
  \leavevmode
  \hbox{
    \begin{beamercolorbox}[wd=.76\paperwidth,ht=2.8ex,dp=1.1ex,leftskip=1em,rightskip=1em]{palette secondary}
      \usebeamerfont{author in head/foot}\insertshorttitle
    \end{beamercolorbox}
    \begin{beamercolorbox}[wd=.24\paperwidth,ht=2.8ex,dp=1.1ex,leftskip=.6em,rightskip=.8em]{palette primary}
      \hfill\insertframenumber{} / \inserttotalframenumber
    \end{beamercolorbox}
  }
}

\setbeamertemplate{title page}{
  \vspace*{0.18\paperheight}
  \begin{beamercolorbox}[wd=\paperwidth,sep=1.4em,leftskip=1.2em,rightskip=1.2em]{title}
    {\usebeamerfont{title}\inserttitle\par}
    \vspace{0.8em}
    {\color{TalexaGold}\rule{0.42\paperwidth}{1.2pt}\par}
  \end{beamercolorbox}
  \vfill
}
"""

    def apply_visual_design(self, code):
        if not isinstance(code, str):
            return code
        if "% Talexa academic visual design" in code:
            return code

        match = re.search(
            r'(\\documentclass(?:\[[^\]]*\])?\{beamer\})(.*?)(\\begin\{document\})',
            code,
            flags=re.DOTALL,
        )
        if not match:
            return code

        documentclass = match.group(1)
        preamble = match.group(2).rstrip()
        begin_document = match.group(3)
        design_block = self.academic_design_block().strip()

        new_preamble_parts = [documentclass]
        if preamble:
            new_preamble_parts.append(preamble)
        new_preamble_parts.append(design_block)
        new_preamble = "\n\n".join(new_preamble_parts)

        return code[:match.start()] + new_preamble + "\n\n" + begin_document + code[match.end():]

    def _chat_with_ollama(self, model_name, messages, request_label, options=None):
        try:
            response = self.ollama_client.chat(
                model=model_name,
                messages=messages,
                options=options,
            )
        except httpx.TimeoutException as exc:
            raise RuntimeError(
                f"Ollama timed out after {self.ollama_timeout:g}s while {request_label} "
                f"with model '{model_name}'. Increase OLLAMA_TIMEOUT, use a smaller/faster model, "
                "or verify the Ollama server is healthy."
            ) from exc
        except ConnectionError as exc:
            raise RuntimeError(
                f"Could not reach Ollama while {request_label}. "
                "Make sure the Ollama app/server is running and accessible."
            ) from exc
        except Exception as exc:
            raise RuntimeError(
                f"Ollama request failed while {request_label} with model '{model_name}': {exc}"
            ) from exc

        return response["message"]["content"]

    def query_ollama(self, model_name, system_prompt, user_prompt, options=None):
        return self._chat_with_ollama(
            model_name=model_name,
            request_label="generating slide content",
            options=options,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )

    def query_ollama_with_image(self, model_name, system_prompt, user_prompt, image_path, options=None):
        return self._chat_with_ollama(
            model_name=model_name,
            request_label=f"reviewing slide image '{image_path}'",
            options=options,
            messages=[
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": user_prompt,
                    "images": [str(image_path)],
                },
            ],
        )

    def compile_tex(self, tex_path):
        tex_path = Path(tex_path).resolve()
        if not tex_path.exists():
            raise FileNotFoundError(f"Tex file {tex_path} does not exist")

        candidates = [
            ("tectonic", ["tectonic", str(tex_path)]),
            ("pdflatex", ["pdflatex", "-interaction=nonstopmode", str(tex_path.name)]),
            ("xelatex", ["xelatex", "-interaction=nonstopmode", str(tex_path.name)]),
            ("lualatex", ["lualatex", "-interaction=nonstopmode", str(tex_path.name)]),
        ]
        available = [(name, cmd) for name, cmd in candidates if shutil.which(name)]

        if not available:
            raise FileNotFoundError(
                "No LaTeX compiler was found in PATH. Install Tectonic or a TeX engine like pdflatex/xelatex/lualatex."
            )

        compiler_name, command = available[0]
        run_kwargs = {
            "capture_output": True,
            "text": True,
        }
        if compiler_name != "tectonic":
            run_kwargs["cwd"] = str(tex_path.parent)

        try:
            result = subprocess.run(
                command,
                check=True,
                **run_kwargs,
            )
            print(f"Tex file compilation succeeded with {compiler_name}.")
            return "\n".join([result.stdout, result.stderr])
        except subprocess.CalledProcessError as e:
            print(f"Compilation failed with {compiler_name}:")
            print(e.stderr)
            return e.stderr or e.stdout

    def correcte_error(self, beamer_code, error_info):
        with open(self.correct_prompt_path, "r", encoding="utf-8") as f:
            template_prompt = f.read()

        inference_prompt = "\n".join([
            template_prompt,
            "This is the latex code for slides:",
            beamer_code,
            "The errors are:",
            "\n".join(error_info),
        ])

        content = self.query_ollama(
            model_name=self.llm_model,
            system_prompt="You are a LaTeX Beamer error correction assistant.",
            user_prompt=inference_prompt,
        )

        code = self.extract_beamer_code(content)
        return self.apply_visual_design(code)

    def pdf2img(self, pdf_path, image_dir, dpi=300, fmt="png", strict_single_page=True):
        pdf_path = Path(pdf_path)
        image_dir = Path(image_dir)

        if pdf_path.suffix.lower() != ".pdf":
            raise ValueError(f"not pdf file: {pdf_path}")
        if not pdf_path.exists():
            raise FileNotFoundError(f"can not find: {pdf_path}")

        with fitz.open(pdf_path) as doc:
            if strict_single_page and doc.page_count != 1:
                raise ValueError(f"not single slide {doc.page_count}: {pdf_path}")

            page = doc[0]
            scale = dpi / 72.0
            mat = fitz.Matrix(scale, scale)
            pix = page.get_pixmap(matrix=mat, alpha=False)

        image_dir.mkdir(parents=True, exist_ok=True)
        fmt = fmt.lower()
        if fmt == "jpeg":
            fmt = "jpg"

        out_path = image_dir / f"{pdf_path.stem}.{fmt}"
        pix.save(out_path.as_posix())
        return out_path

    def render_pdf_pages(self, pdf_path, image_dir, dpi=200, fmt="png"):
        pdf_path = Path(pdf_path)
        image_dir = Path(image_dir)

        if pdf_path.suffix.lower() != ".pdf":
            raise ValueError(f"not pdf file: {pdf_path}")
        if not pdf_path.exists():
            raise FileNotFoundError(f"can not find: {pdf_path}")

        image_dir.mkdir(parents=True, exist_ok=True)
        fmt = fmt.lower()
        if fmt == "jpeg":
            fmt = "jpg"

        saved_paths = []

        with fitz.open(pdf_path) as doc:
            scale = dpi / 72.0
            mat = fitz.Matrix(scale, scale)

            for i, page in enumerate(doc):
                pix = page.get_pixmap(matrix=mat, alpha=False)
                out_path = image_dir / f"{pdf_path.stem}_page_{i+1:03d}.{fmt}"
                pix.save(out_path.as_posix())
                saved_paths.append(str(out_path))

        return saved_paths

    def add_small_after_blocks(self, tex) -> str:
        pattern = re.compile(
            r'(?m)^([ \t]*)\\begin\{(?:block|alertblock|exampleblock)\}'
            r'(?:<[^>\n]*>)?(?:\[[^\]\n]*\])?\s*\{[^}]*\}[^\n]*\r?\n'
            r'([ \t]*)(?!\\small\b)'
        )

        def repl(m: re.Match) -> str:
            return f"{m.group(0)}\\footnotesize\n{m.group(2)}"

        return pattern.sub(repl, tex)

    def scale_includegraphics_widths(self, tex: str, factor: float, precision: int = 3, add_if_missing: bool = False) -> str:
        include_re = re.compile(
            r'\\includegraphics(?:\s*\[(?P<opts>[^\]]*)\])?\s*\{(?P<path>[^}]*)\}',
            re.DOTALL,
        )
        width_re = re.compile(r'(?<![a-zA-Z])width\s*=\s*([^,\]]+)', re.IGNORECASE)
        rel_re = re.compile(r'^\s*(?:(\d*\.?\d+)|\.(\d+))?\s*\\(textwidth|linewidth|columnwidth)\b')

        def scale_rel(expr: str):
            val = expr.strip().strip("{}")
            m = rel_re.match(val)
            if not m:
                return None
            num = m.group(1)
            if num is None and m.group(2) is not None:
                num = "0." + m.group(2)
            k = 1.0 if not num else float(num)
            new_k = round(k * factor, precision)
            return f"{new_k:g}\\{m.group(3)}"

        def repl_inc(mm: re.Match):
            opts = mm.group("opts")
            img_path = mm.group("path")

            if opts is None or opts.strip() == "":
                if add_if_missing:
                    return f"\\includegraphics[width={factor:g}\\textwidth]{{{img_path}}}"
                return mm.group(0)

            def repl_width(mw: re.Match):
                expr = mw.group(1)
                scaled = scale_rel(expr)
                return f"width={scaled}" if scaled is not None else mw.group(0)

            new_opts = width_re.sub(repl_width, opts)
            if new_opts == opts and add_if_missing:
                new_opts = f"width={factor:g}\\textwidth," + opts.strip()

            return f"\\includegraphics[{new_opts}]{{{img_path}}}"

        return include_re.sub(repl_inc, tex)

    def _line_starts(self, text):
        starts = [0]
        for m in re.finditer('\n', text):
            starts.append(m.end())
        return starts

    def _pos_to_line(self, pos, line_starts):
        return bisect.bisect_right(line_starts, pos)

    def compute_frame_spans(self, code: str):
        line_starts = self._line_starts(code)

        sec_re = re.compile(r'(?m)^\\section\*?(?:\[[^\]]*\])?\{([^}]*)\}')
        sub_re = re.compile(r'(?m)^\\subsection\*?(?:\[[^\]]*\])?\{([^}]*)\}')

        sections = []
        for m in sec_re.finditer(code):
            pos = m.start()
            sections.append({
                "pos": pos,
                "line": self._pos_to_line(pos, line_starts),
                "title": m.group(1).strip()
            })

        subsections = []
        for m in sub_re.finditer(code):
            pos = m.start()
            subsections.append({
                "pos": pos,
                "line": self._pos_to_line(pos, line_starts),
                "title": m.group(1).strip()
            })

        sec_pos_list = [s["pos"] for s in sections]
        sub_pos_list = [s["pos"] for s in subsections]

        frame_re = re.compile(
            r'\\begin\{frame\}(?:<[^>\n]*>)?(?:\[[^\]\n]*\])?(?:\{.*?\}){0,2}.*?\\end\{frame\}',
            re.DOTALL
        )
        frametitle_re = re.compile(r'\\frametitle(?:<[^>]*>)?(?:\[[^\]]*\])?\{([^}]*)\}')
        frame_env_title_re = re.compile(
            r'^\\begin\{frame\}(?:<[^>\n]*>)?(?:\[[^\]\n]*\])?\s*\{([^}]*)\}',
            re.DOTALL
        )

        frames = []
        for i, m in enumerate(frame_re.finditer(code)):
            start, end = m.start(), m.end()
            start_line = self._pos_to_line(start, line_starts)
            end_line = self._pos_to_line(end - 1, line_starts)
            text = m.group(0)

            t = frametitle_re.search(text)
            if t:
                title = t.group(1).strip()
            else:
                t2 = frame_env_title_re.search(text)
                title = t2.group(1).strip() if t2 else ""

            if sec_pos_list:
                j = bisect_right(sec_pos_list, start) - 1
                if j >= 0:
                    sec_title = sections[j]["title"]
                    sec_line = sections[j]["line"]
                else:
                    sec_title, sec_line = "", None
            else:
                sec_title, sec_line = "", None

            if sub_pos_list:
                k = bisect_right(sub_pos_list, start) - 1
                if k >= 0:
                    sub_title = subsections[k]["title"]
                    sub_line = subsections[k]["line"]
                else:
                    sub_title, sub_line = "", None
            else:
                sub_title, sub_line = "", None

            frames.append({
                "idx": i,
                "start": start,
                "end": end,
                "start_line": start_line,
                "end_line": end_line,
                "title": title,
                "section": sec_title,
                "section_line": sec_line,
                "subsection": sub_title,
                "subsection_line": sub_line,
                "text": text
            })

        return frames

    def make_grid_with_labels(
        self,
        img_paths: Sequence[str],
        out_path: str,
        cell_size: Tuple[int, int] = (512, 512),
        gap: int = 16,
        rows: int = 2,
        cols: int = 2,
        labels: Optional[Sequence[str]] = None,
        bg_color: Tuple[int, int, int] = (255, 255, 255),
        font_path: Optional[str] = None,
        font_size: Optional[int] = None,
    ) -> Path:
        n = rows * cols
        if len(img_paths) != n:
            raise ValueError(f"img_paths must contain {n} image paths (got {len(img_paths)})")

        if labels is None:
            labels = list(string.ascii_uppercase[:n])

        cw, ch = cell_size
        canvas_w = cw * cols + gap * (cols - 1)
        canvas_h = ch * rows + gap * (rows - 1)
        canvas = Image.new("RGB", (canvas_w, canvas_h), bg_color)

        def _to_rgb(img: Image.Image) -> Image.Image:
            if img.mode in ("RGBA", "LA"):
                base = Image.new("RGB", img.size, bg_color)
                base.paste(img, mask=img.split()[-1])
                return base
            return img.convert("RGB")

        if font_size is None:
            font_size = max(16, int(min(cw, ch) * 0.08))

        font = None
        if font_path:
            try:
                font = ImageFont.truetype(font_path, font_size)
            except Exception:
                font = None

        if font is None:
            for try_name in ["DejaVuSans-Bold.ttf", "Arial.ttf", "Helvetica.ttf"]:
                try:
                    font = ImageFont.truetype(try_name, font_size)
                    break
                except Exception:
                    continue

        if font is None:
            font = ImageFont.load_default()

        draw = ImageDraw.Draw(canvas)

        positions = []
        for r in range(rows):
            for c in range(cols):
                x0 = c * (cw + gap)
                y0 = r * (ch + gap)
                positions.append((x0, y0))

        for i, (p, (x0, y0)) in enumerate(zip(img_paths, positions)):
            with Image.open(p) as im_raw:
                im = _to_rgb(im_raw)

            w, h = im.size
            scale = min(cw / w, ch / h)
            nw, nh = max(1, int(w * scale)), max(1, int(h * scale))
            im_resized = im.resize((nw, nh), Image.BICUBIC)

            px = x0 + (cw - nw) // 2
            py = y0 + (ch - nh) // 2
            canvas.paste(im_resized, (px, py))

            label = labels[i]
            margin = max(6, font_size // 4)
            tx, ty = x0 + margin, y0 + margin
            draw.text(
                (tx, ty),
                label,
                font=font,
                fill=(255, 0, 0),
                stroke_width=max(1, font_size // 16),
                stroke_fill=(255, 0, 0)
            )

        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        canvas.save(out_path.as_posix())
        return out_path

    def improve_layout(self, code, feedback, beamer_save_path):
        with open(self.select_proposal_prompt_path, "r", encoding="utf-8") as f:
            template_prompt = f.read()

        warning_info = re.findall(r'^(warning: .+)', feedback, flags=re.MULTILINE)
        warning_info = warning_info[:len(warning_info)//2]
        warning_info = [s for s in warning_info if 'Overfull' in s]

        head_match = re.search(
            r'\\documentclass(?:\[[^\]]*\])?\{beamer\}(.*?)\\begin{document}',
            code,
            flags=re.DOTALL
        )
        if not head_match:
            return beamer_save_path.replace(".tex", ".pdf")

        head = head_match.group(1)
        head = head + "\n" + "\\setbeamerfont{caption}{size=\\scriptsize}"

        frames = self.compute_frame_spans(code)
        need_improve_list = []

        for warning in warning_info:
            line_match = re.search(r'(?<=\.tex:)\d+', warning)
            if not line_match:
                continue

            num = int(line_match.group())
            for idx, f in enumerate(frames):
                if f["start_line"] <= num <= f["end_line"]:
                    if "\\includegraphics" in f["text"]:
                        need_improve_list.append(idx)
                    break

        need_improve_list = sorted(set(need_improve_list))

        proposal_tmp_dir = path.join("Data", "intermediate", "proposal_imgs")
        os.makedirs(proposal_tmp_dir, exist_ok=True)

        factors = [1, 0.75, 0.5, 0.25]
        map_dic = {"A": 0, "B": 1, "C": 2, "D": 3}

        for frame_idx in need_improve_list:
            frame = frames[frame_idx]
            proposal_imgs_path_list = []
            proposal_code_list = []

            for factor in factors:
                proposal_code = self.scale_includegraphics_widths(frame["text"], factor)
                proposal_code = self.add_small_after_blocks(proposal_code)
                proposal_full_code = "\n".join([
                    "\\documentclass{beamer}",
                    head,
                    "\\begin{document}",
                    proposal_code,
                    "\\end{document}"
                ])

                proposal_code_save_path = beamer_save_path.replace(".tex", f"_proposal_{factor}.tex")

                with open(proposal_code_save_path, "w", encoding="utf-8") as f:
                    f.write(proposal_full_code)

                self.compile_tex(proposal_code_save_path)
                img_path = self.pdf2img(
                    proposal_code_save_path.replace(".tex", ".pdf"),
                    proposal_tmp_dir
                )
                proposal_imgs_path_list.append(str(img_path))
                proposal_code_list.append(proposal_code)

            prompt_img_path = path.join(proposal_tmp_dir, "merged.png")
            self.make_grid_with_labels(
                proposal_imgs_path_list,
                prompt_img_path,
                rows=2,
                cols=2
            )

            content = self.query_ollama_with_image(
                model_name=self.vlm_model,
                system_prompt='You are a slide layout selection assistant. Respond only in JSON like {"choice":"A"}.',
                user_prompt="\n".join([template_prompt, "Here are the choices A, B, C, D"]),
                image_path=prompt_img_path,
            )

            choice_str = self.extract_json_block(content)
            try:
                choice = json.loads(choice_str)
                refined_code = proposal_code_list[map_dic[choice["choice"]]]
            except Exception:
                refined_code = proposal_code_list[0]

            frames[frame_idx]["text"] = refined_code

        new_code = ["\\documentclass{beamer}", head, "\\begin{document}"]
        section = []
        subsection = []

        for frame in frames:
            if len(frame["section"]) != 0 and frame["section"] not in section:
                new_code.append("\\section{{{}}}".format(frame["section"]))
                section.append(frame["section"])
                subsection = []

            if len(frame["subsection"]) != 0 and frame["subsection"] not in subsection:
                new_code.append("\\subsection{{{}}}".format(frame["subsection"]))
                subsection.append(frame["subsection"])

            new_code.append(self.add_small_after_blocks(frame["text"]))

        new_code.append("\\end{document}")
        new_code = "\n".join(new_code)

        new_code_save_path = beamer_save_path.replace(".tex", "_refined.tex")
        with open(new_code_save_path, "w", encoding="utf-8") as f:
            f.write(new_code)

        self.compile_tex(new_code_save_path)
        return new_code_save_path.replace(".tex", ".pdf")

    def generate_beamer(self, latex_input_path, beamer_save_path, beamer_temp_name=None):
        if not os.path.exists(latex_input_path):
            raise FileNotFoundError(f"Latex input file not found: {latex_input_path}")

        with open(self.slide_prompt_path, "r", encoding="utf-8") as f:
            template_prompt = f.read()

        with open(latex_input_path, "r", encoding="utf-8") as f:
            tex_content = f.read()

        if beamer_temp_name is None:
            main_inference_prompt = [
                template_prompt,
                "This is the lecture/source content to convert into Beamer slides:",
                tex_content,
            ]
        else:
            main_inference_prompt = [
                template_prompt,
                "This is the lecture/source content to convert into Beamer slides:",
                tex_content,
                "Use Beamer Theme: {}".format(beamer_temp_name)
            ]
        main_inference_prompt = "\n".join(map(str, main_inference_prompt))

        content = self.query_ollama(
            model_name=self.llm_model,
            system_prompt="You are a Beamer slide generation assistant.",
            user_prompt=main_inference_prompt,
            options={"temperature": 0.2},
        )

        code = self.extract_beamer_code(content)
        code = self.apply_visual_design(code)
        if not isinstance(code, str):
            print("Failed to generate beamer code.")
            print(content)
            return None

        with open(beamer_save_path, "w", encoding="utf-8") as f:
            f.write(code)

        return code

    def cleanup_final_outputs(self, final_pdf, primary_tex_path=None, extra_dirs=None):
        final_pdf = Path(final_pdf).resolve()
        final_tex = final_pdf.with_suffix(".tex")
        output_dir = final_pdf.parent

        prefixes = {final_pdf.stem}
        if primary_tex_path is not None:
            primary_stem = Path(primary_tex_path).resolve().stem
            prefixes.add(primary_stem)
            prefixes.add(f"{primary_stem}_refined")

        keep_paths = {final_pdf, final_tex}
        removable_suffixes = [
            ".aux", ".log", ".nav", ".out", ".snm", ".toc", ".vrb",
            ".fls", ".fdb_latexmk", ".synctex.gz", ".pdf", ".tex",
        ]

        for item in output_dir.iterdir():
            if not item.is_file():
                continue
            if item.resolve() in keep_paths:
                continue

            item_name = item.name
            if any(item_name.startswith(prefix) for prefix in prefixes):
                if any(item_name.endswith(suffix) for suffix in removable_suffixes):
                    try:
                        item.unlink()
                    except FileNotFoundError:
                        pass

        for extra_dir in extra_dirs or []:
            extra_path = Path(extra_dir)
            if extra_path.exists():
                shutil.rmtree(extra_path, ignore_errors=True)

    def run(
        self,
        latex_input_path,
        beamer_save_path,
        beamer_temp_name=None,
        max_fix_attempts=10,
        improve=True,
        intermediate_image_dir=None,
    ):
        print("Starting SlideBuilderAgent...\n")

        beamer_save_path = str(Path(beamer_save_path).resolve())
        Path(beamer_save_path).parent.mkdir(parents=True, exist_ok=True)

        code = self.generate_beamer(
            latex_input_path=latex_input_path,
            beamer_save_path=beamer_save_path,
            beamer_temp_name=beamer_temp_name,
        )

        if code is None:
            print("Slide generation failed.")
            return None

        feedback = self.compile_tex(beamer_save_path)

        attempt = 0
        while attempt < max_fix_attempts:
            if "error" in feedback.lower():
                print(f"\nFix attempt {attempt + 1} of {max_fix_attempts}")
                error_info = re.findall(r'^(error: .+)', feedback, flags=re.MULTILINE)

                fixed_code = self.correcte_error(code, error_info)
                if not isinstance(fixed_code, str):
                    print("Failed to fix code.")
                    break

                code = fixed_code

                with open(beamer_save_path, "w", encoding="utf-8") as f:
                    f.write(code)

                feedback = self.compile_tex(beamer_save_path)
                attempt += 1
            else:
                break

        if improve:
            final_pdf = self.improve_layout(code, feedback, beamer_save_path)
        else:
            final_pdf = beamer_save_path.replace(".tex", ".pdf")

        if intermediate_image_dir is None:
            raise ValueError("intermediate_image_dir must be provided")

        rendered_images = self.render_pdf_pages(final_pdf, intermediate_image_dir, dpi=200)

        print("\nRendered slide images:")
        for img in rendered_images:
            print(img)

        self.cleanup_final_outputs(
            final_pdf=final_pdf,
            primary_tex_path=beamer_save_path,
            extra_dirs=[path.join("Data", "intermediate", "proposal_imgs")],
        )

        print("\nSlideBuilderAgent completed successfully.")
        print(f"Final PDF: {final_pdf}")
        return final_pdf


if __name__ == "__main__":
    agent = SlideBuilderAgent(
        llm_model="qwen2.5:7b",
        vlm_model="qwen2.5:7b"
    )

    final_pdf = agent.run(
        latex_input_path="Data/output/lecture1.tex",
        beamer_save_path="lecture1_slides.tex",
        beamer_temp_name=None,
        max_fix_attempts=10,
        improve=True,
        intermediate_image_dir="Data/intermediate/lecture1_slides_refined",
    )

    print("\nGenerated file:", final_pdf)
