from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from PIPELINE.langchain_pipeline import create_slides_pipeline
from PIPELINE.session_manager import SessionPaths


def run_slides_pipeline(
    session: SessionPaths,
    lecture_language: str,
) -> dict[str, object]:
    pipeline = create_slides_pipeline(
        model_name="qwen2.5:7b",
        base_data_dir=session.session_dir,
    )

    return pipeline.invoke(
        {
            "source_type": "slides",
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
