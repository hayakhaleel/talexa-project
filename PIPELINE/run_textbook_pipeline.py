from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from PIPELINE.langchain_pipeline import create_textbook_pipeline
from PIPELINE.session_manager import SessionPaths


def run_textbook_pipeline(
    session: SessionPaths,
    lecture_language: str,
    model_name: str = "qwen2.5:7b",
) -> dict[str, object]:
    pipeline = create_textbook_pipeline(
        model_name=model_name,
        base_data_dir=session.session_dir,
    )

    # LangChain invokes the textbook flow using a shared state dictionary.
    return pipeline.invoke(
        {
            "source_type": "textbook",
            "language": lecture_language,
            "session_number": session.session_number,
            "session_dir": str(session.session_dir),
            "input_dir": str(session.input_dir),
            "intermediate_dir": str(session.intermediate_dir),
            "output_dir": str(session.output_dir),
            "input_path": str(session.stored_pdf_path),
            "audio_path": str(session.stored_audio_path) if session.stored_audio_path else None,
            "portrait_image_path": (
                str(session.stored_portrait_path) if session.stored_portrait_path else None
            ),
        }
    )
