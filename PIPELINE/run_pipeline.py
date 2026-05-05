from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from PIPELINE.session_manager import create_session
from PIPELINE.preprocessing import preprocess_inputs
from PIPELINE.assembly import assemble_session_video
from PIPELINE.run_slides_pipeline import run_slides_pipeline
from PIPELINE.run_textbook_pipeline import run_textbook_pipeline
from PIPELINE.session_manager import SessionPaths


def _ask_source_type() -> str:
    while True:
        choice = input(
            "Generate lectures from a textbook chapter or slides? "
            "Type 'textbook' or 'slides': "
        ).strip().lower()

        if choice in {"textbook", "slides"}:
            return choice

        print("Please type either 'textbook' or 'slides'.")


def _ask_language() -> str:
    while True:
        choice = input(
            "Which lecture language do you want? Type 'english' or 'arabic': "
        ).strip().lower()

        if choice in {"english", "arabic"}:
            return choice

        print("Please type either 'english' or 'arabic'.")


def run_pipeline_for_session(
    *,
    session: SessionPaths,
    source_type: str,
    lecture_language: str,
    fps: int = 24,
    ffmpeg_path: str | None = None,
) -> dict[str, object]:
    normalized_source_type = source_type.strip().lower()
    normalized_language = lecture_language.strip().lower()

    if normalized_source_type == "textbook":
        result = run_textbook_pipeline(session, lecture_language=normalized_language)
    elif normalized_source_type == "slides":
        result = run_slides_pipeline(session, lecture_language=normalized_language)
    else:
        raise ValueError("source_type must be either 'textbook' or 'slides'.")

    assembly_result = assemble_session_video(
        session_id=session.session_number,
        language=normalized_language,
        fps=fps,
        ffmpeg_path=ffmpeg_path,
    )

    merged_result = dict(result)
    merged_result.update(assembly_result)
    return merged_result


def main() -> None:
    source_type = _ask_source_type()
    lecture_language = _ask_language()
    pdf_path, audio_path, portrait_path = preprocess_inputs(source_type)
    session = create_session(
        PROJECT_ROOT,
        pdf_file_path=pdf_path,
        audio_file_path=audio_path,
        portrait_file_path=portrait_path,
    )

    print(f"Created session {session.session_number} at: {session.session_dir}")
    print(f"Requested language: {lecture_language}")
    print(f"Stored PDF at: {session.stored_pdf_path}")
    print(f"Stored audio at: {session.stored_audio_path}")
    print(f"Stored portrait at: {session.stored_portrait_path}")

    result = run_pipeline_for_session(
        session=session,
        source_type=source_type,
        lecture_language=lecture_language,
    )

    if source_type == "textbook":
        print("\nTextbook pipeline completed successfully.")
        print(f"Summary file: {result['summary_path']}")
        print(f"LaTeX file: {result['latex_path']}")
        print(f"Slides PDF: {result['slides_pdf_path']}")
        print(f"Slide images folder: {result['slide_images_dir']}")
        print(f"Lecture video: {result['video_output_path']}")
        return

    print("\nSlides pipeline completed successfully.")
    print(f"LaTeX file: {result['latex_path']}")
    print(f"Slides PDF: {result['slides_pdf_path']}")
    print(f"Slide images folder: {result['slide_images_dir']}")
    print(f"Lecture video: {result['video_output_path']}")


if __name__ == "__main__":
    main()
