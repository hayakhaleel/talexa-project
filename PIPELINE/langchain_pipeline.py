from __future__ import annotations

from pathlib import Path
from typing import Any

try:
    from langchain_core.runnables import RunnableLambda
except ImportError as exc:
    raise ImportError(
        "LangChain is required for the PIPELINE orchestration. "
        "Install a runtime with 'langchain-core' available before running this pipeline."
    ) from exc

from Agents.summary_agent import SummaryAgent
from Agents.Slider_Builder_Agent import SlideBuilderAgent
from Agents.Textbook_latex_agent import LatexAgent
from Agents.Translation_agent import JsonTranslateAgent
from Agents.cursor_agent import CursorAgent
from Agents.slides_latex_agent import LatexAgent as SlidesLatexAgent
from Agents.speech_agent import SpeechAgent
from Agents.subtitle_focus_agent import SubtitleFocusAgent
from Agents.talking_head_agent import TalkingHeadApiAgent

DEFAULT_TRANSLATION_MODEL = "facebook/nllb-200-distilled-600M"


def _file_stem_from_state(state: dict[str, Any]) -> str:
    return Path(state["input_path"]).stem


def build_textbook_pipeline(
    summary_agent: SummaryAgent,
    latex_agent: LatexAgent,
    slide_builder_agent: SlideBuilderAgent,
    subtitle_agent: SubtitleFocusAgent,
    translation_agent: JsonTranslateAgent,
):
    def run_summary(state: dict[str, Any]) -> dict[str, Any]:
        input_path = Path(state["input_path"])
        summary_output_path = Path(state["intermediate_dir"]) / f"SUMMARY_{input_path.stem}.txt"

        summary_path = summary_agent.run(
            pdf_path=str(input_path),
            output_txt_path=str(summary_output_path),
            max_pages=None,
        )

        updated_state = dict(state)
        updated_state["summary_path"] = summary_path
        return updated_state

    def run_latex_conversion(state: dict[str, Any]) -> dict[str, Any]:
        input_path = Path(state["input_path"])
        latex_output_path = Path(state["intermediate_dir"]) / f"LATEX_{input_path.stem}.tex"

        latex_path = latex_agent.run(
            summary_path=state["summary_path"],
            output_tex_path=str(latex_output_path),
            compile_pdf=False,
        )

        updated_state = dict(state)
        updated_state["latex_path"] = latex_path
        return updated_state

    def run_slide_builder(state: dict[str, Any]) -> dict[str, Any]:
        input_path = Path(state["input_path"])
        slide_base_name = f"SLIDES_{input_path.stem}"
        slide_builder_tex_path = Path(state["output_dir"]) / f"{slide_base_name}.tex"
        slide_images_dir = Path(state["intermediate_dir"]) / f"{slide_base_name}_refined"

        raw_pdf_path = slide_builder_agent.run(
            latex_input_path=state["latex_path"],
            beamer_save_path=str(slide_builder_tex_path),
            beamer_temp_name=None,
            max_fix_attempts=10,
            improve=True,
            intermediate_image_dir=str(slide_images_dir),
        )

        final_pdf_target = Path(state["output_dir"]) / f"{slide_base_name}.pdf"
        raw_pdf_path = Path(raw_pdf_path)

        if raw_pdf_path.resolve() != final_pdf_target.resolve():
            if final_pdf_target.exists():
                final_pdf_target.unlink()
            raw_pdf_path.replace(final_pdf_target)

        updated_state = dict(state)
        updated_state["slides_pdf_path"] = str(final_pdf_target)
        updated_state["slide_images_dir"] = str(slide_images_dir)
        return updated_state

    def run_subtitle_builder(state: dict[str, Any]) -> dict[str, Any]:
        input_stem = _file_stem_from_state(state)
        subtitle_output_path = Path(state["intermediate_dir"]) / f"SUBTITLE_{input_stem}.json"

        subtitle_agent.output_path = str(subtitle_output_path)
        subtitle_agent.run(state["slide_images_dir"])

        updated_state = dict(state)
        updated_state["subtitle_path"] = str(subtitle_output_path)
        return updated_state

    def run_translation_if_needed(state: dict[str, Any]) -> dict[str, Any]:
        updated_state = dict(state)
        language = str(state.get("language", "english")).lower()

        if language == "english":
            updated_state["translation_path"] = None
            updated_state["speech_input_path"] = state["subtitle_path"]
            return updated_state

        input_stem = _file_stem_from_state(state)
        translation_output_path = Path(state["intermediate_dir"]) / f"TRANSLATION_{input_stem}.json"

        translation_path = translation_agent.run(
            input_json_path=state["subtitle_path"],
            output_json_path=str(translation_output_path),
        )

        updated_state["translation_path"] = translation_path
        updated_state["speech_input_path"] = translation_path
        return updated_state

    def run_speech(state: dict[str, Any]) -> dict[str, Any]:
        input_stem = _file_stem_from_state(state)
        normalized_language = str(state.get("language", "english")).lower()
        language = "ar" if normalized_language == "arabic" else "en"
        audio_output_dir = Path(state["intermediate_dir"]) / f"AUDIO_{normalized_language.upper()}_{input_stem}"

        if normalized_language == "english":
            speech_input_path = state["subtitle_path"]
        elif normalized_language == "arabic":
            translation_path = state.get("translation_path")
            if not translation_path:
                raise ValueError("Arabic speech requires translated subtitles, but translation_path is missing.")
            speech_input_path = translation_path
        else:
            raise ValueError("language must be either 'english' or 'arabic'.")

        print(f"[Pipeline] Speech input ({normalized_language}): {speech_input_path}")

        speech_agent = SpeechAgent(
            subtitles_json_path=speech_input_path,
            ref_audio_path=state["audio_path"],
            output_dir=str(audio_output_dir),
        )
        speech_agent.run()

        updated_state = dict(state)
        updated_state["audio_output_dir"] = str(audio_output_dir)
        updated_state["speech_input_path"] = speech_input_path
        return updated_state

    def run_cursor(state: dict[str, Any]) -> dict[str, Any]:
        input_stem = _file_stem_from_state(state)
        cursor_output_path = Path(state["intermediate_dir"]) / f"CURSOR_{input_stem}.json"
        cursor_input_path = state["speech_input_path"]

        cursor_agent = CursorAgent(
            model="qwen2.5:7b",
            images_dir=state["slide_images_dir"],
            audio_dir=state["audio_output_dir"],
        )
        cursor_agent.run(
            input_json_path=cursor_input_path,
            output_json_path=str(cursor_output_path),
        )

        updated_state = dict(state)
        updated_state["cursor_path"] = str(cursor_output_path)
        return updated_state

    def run_talking_head(state: dict[str, Any]) -> dict[str, Any]:
        input_stem = _file_stem_from_state(state)
        talking_head_output_dir = Path(state["intermediate_dir"]) / f"TALKING_HEAD_{input_stem}"

        talking_head_agent = TalkingHeadApiAgent(
            source_image=state["portrait_image_path"],
            audio_dir=state["audio_output_dir"],
            output_dir=str(talking_head_output_dir),
            api_key=state.get("heygen_api_key"),
            title=f"Talexa_{input_stem}",
        )

        result = talking_head_agent.run()

        updated_state = dict(state)
        updated_state["talking_head_output_dir"] = str(talking_head_output_dir)
        updated_state["talking_head_video_path"] = result["final_video_path"]
        updated_state["talking_head_video_id"] = result["video_id"]
        updated_state["talking_head_video_url"] = result["video_url"]
        return updated_state

    return (
        RunnableLambda(run_summary)
        | RunnableLambda(run_latex_conversion)
        | RunnableLambda(run_slide_builder)
        | RunnableLambda(run_subtitle_builder)
        | RunnableLambda(run_translation_if_needed)
        | RunnableLambda(run_speech)
        | RunnableLambda(run_cursor)
        | RunnableLambda(run_talking_head)
    )


def build_slides_pipeline(
    slides_latex_agent: SlidesLatexAgent,
    subtitle_agent: SubtitleFocusAgent,
    translation_agent: JsonTranslateAgent,
):
    def run_slides_latex(state: dict[str, Any]) -> dict[str, Any]:
        input_path = Path(state["input_path"])
        input_stem = input_path.stem
        latex_base_name = f"LATEX_{input_stem}"
        latex_output_dir = Path(state["intermediate_dir"])

        raw_pdf_path = slides_latex_agent.run(
            input_pdf_path=str(input_path),
            output_filename=latex_base_name,
            output_dir=str(latex_output_dir),
        )

        if raw_pdf_path is None:
            raise RuntimeError("Slides LaTeX agent failed to generate a PDF.")

        raw_pdf_path = Path(raw_pdf_path)
        latex_path = latex_output_dir / f"{latex_base_name}.tex"
        slides_pdf_path = Path(state["output_dir"]) / f"SLIDES_{input_stem}.pdf"
        slide_images_dir = Path(state["intermediate_dir"]) / f"SLIDES_IMAGES_{input_stem}"
        renderer = SlideBuilderAgent()

        if latex_path.exists():
            latex_code = latex_path.read_text(encoding="utf-8")
            designed_latex_code = renderer.apply_visual_design(latex_code)
            if designed_latex_code != latex_code:
                latex_path.write_text(designed_latex_code, encoding="utf-8")
                designed_feedback = renderer.compile_tex(str(latex_path))
                designed_pdf_path = latex_path.with_suffix(".pdf")
                if designed_pdf_path.exists() and not renderer.has_fatal_latex_error(designed_feedback):
                    raw_pdf_path = designed_pdf_path

        if raw_pdf_path.resolve() != slides_pdf_path.resolve():
            if slides_pdf_path.exists():
                slides_pdf_path.unlink()
            raw_pdf_path.replace(slides_pdf_path)

        renderer.render_pdf_pages(str(slides_pdf_path), str(slide_images_dir), dpi=200)

        updated_state = dict(state)
        updated_state["latex_path"] = str(latex_path)
        updated_state["slides_pdf_path"] = str(slides_pdf_path)
        updated_state["slide_images_dir"] = str(slide_images_dir)
        return updated_state

    def run_subtitle_builder(state: dict[str, Any]) -> dict[str, Any]:
        input_stem = _file_stem_from_state(state)
        subtitle_output_path = Path(state["intermediate_dir"]) / f"SUBTITLE_{input_stem}.json"

        subtitle_agent.output_path = str(subtitle_output_path)
        subtitle_agent.run(state["slide_images_dir"])

        updated_state = dict(state)
        updated_state["subtitle_path"] = str(subtitle_output_path)
        return updated_state

    def run_translation_if_needed(state: dict[str, Any]) -> dict[str, Any]:
        updated_state = dict(state)
        language = str(state.get("language", "english")).lower()

        if language == "english":
            updated_state["translation_path"] = None
            updated_state["speech_input_path"] = state["subtitle_path"]
            return updated_state

        input_stem = _file_stem_from_state(state)
        translation_output_path = Path(state["intermediate_dir"]) / f"TRANSLATION_{input_stem}.json"

        translation_path = translation_agent.run(
            input_json_path=state["subtitle_path"],
            output_json_path=str(translation_output_path),
        )

        updated_state["translation_path"] = translation_path
        updated_state["speech_input_path"] = translation_path
        return updated_state

    def run_speech(state: dict[str, Any]) -> dict[str, Any]:
        input_stem = _file_stem_from_state(state)
        normalized_language = str(state.get("language", "english")).lower()
        language = "ar" if normalized_language == "arabic" else "en"
        audio_output_dir = Path(state["intermediate_dir"]) / f"AUDIO_{normalized_language.upper()}_{input_stem}"

        if normalized_language == "english":
            speech_input_path = state["subtitle_path"]
        elif normalized_language == "arabic":
            translation_path = state.get("translation_path")
            if not translation_path:
                raise ValueError("Arabic speech requires translated subtitles, but translation_path is missing.")
            speech_input_path = translation_path
        else:
            raise ValueError("language must be either 'english' or 'arabic'.")

        print(f"[Pipeline] Speech input ({normalized_language}): {speech_input_path}")

        speech_agent = SpeechAgent(
            subtitles_json_path=speech_input_path,
            ref_audio_path=state["audio_path"],
            output_dir=str(audio_output_dir),
        )
        speech_agent.run()

        updated_state = dict(state)
        updated_state["audio_output_dir"] = str(audio_output_dir)
        updated_state["speech_input_path"] = speech_input_path
        return updated_state

    def run_cursor(state: dict[str, Any]) -> dict[str, Any]:
        input_stem = _file_stem_from_state(state)
        cursor_output_path = Path(state["intermediate_dir"]) / f"CURSOR_{input_stem}.json"

        cursor_agent = CursorAgent(
            model="qwen2.5:7b",
            images_dir=state["slide_images_dir"],
            audio_dir=state["audio_output_dir"],
        )
        cursor_agent.run(
            input_json_path=state["speech_input_path"],
            output_json_path=str(cursor_output_path),
        )

        updated_state = dict(state)
        updated_state["cursor_path"] = str(cursor_output_path)
        return updated_state

    def run_talking_head(state: dict[str, Any]) -> dict[str, Any]:
        input_stem = _file_stem_from_state(state)
        talking_head_output_dir = Path(state["intermediate_dir"]) / f"TALKING_HEAD_{input_stem}"

        talking_head_agent = TalkingHeadApiAgent(
            source_image=state["portrait_image_path"],
            audio_dir=state["audio_output_dir"],
            output_dir=str(talking_head_output_dir),
            api_key=state.get("heygen_api_key"),
            title=f"Talexa_{input_stem}",
        )

        result = talking_head_agent.run()

        updated_state = dict(state)
        updated_state["talking_head_output_dir"] = str(talking_head_output_dir)
        updated_state["talking_head_video_path"] = result["final_video_path"]
        updated_state["talking_head_video_id"] = result["video_id"]
        updated_state["talking_head_video_url"] = result["video_url"]
        return updated_state

    return (
        RunnableLambda(run_slides_latex)
        | RunnableLambda(run_subtitle_builder)
        | RunnableLambda(run_translation_if_needed)
        | RunnableLambda(run_speech)
        | RunnableLambda(run_cursor)
        | RunnableLambda(run_talking_head)
    )


def create_textbook_pipeline(
    model_name: str = "qwen2.5:7b",
    base_data_dir: str | Path | None = None,
):
    summary_agent = SummaryAgent(
        model_name=model_name,
        base_data_dir=str(base_data_dir) if base_data_dir is not None else "Data",
    )
    latex_agent = LatexAgent(model=model_name)
    slide_builder_agent = SlideBuilderAgent(
        llm_model=model_name,
        vlm_model="qwen3-vl:8b",
    )
    subtitle_agent = SubtitleFocusAgent(
        model_name="qwen3-vl:8b",
    )
    translation_agent = JsonTranslateAgent(
        model_name=DEFAULT_TRANSLATION_MODEL,
        base_data_dir=str(base_data_dir) if base_data_dir is not None else "Data",
    )
    return build_textbook_pipeline(
        summary_agent=summary_agent,
        latex_agent=latex_agent,
        slide_builder_agent=slide_builder_agent,
        subtitle_agent=subtitle_agent,
        translation_agent=translation_agent,
    )


def create_slides_pipeline(
    model_name: str = "qwen2.5:7b",
    base_data_dir: str | Path | None = None,
):
    slides_latex_agent = SlidesLatexAgent(model_name=model_name)
    subtitle_agent = SubtitleFocusAgent(
        model_name="qwen3-vl:8b",
    )
    translation_agent = JsonTranslateAgent(
        model_name=DEFAULT_TRANSLATION_MODEL,
        base_data_dir=str(base_data_dir) if base_data_dir is not None else "Data",
    )
    return build_slides_pipeline(
        slides_latex_agent=slides_latex_agent,
        subtitle_agent=subtitle_agent,
        translation_agent=translation_agent,
    )

