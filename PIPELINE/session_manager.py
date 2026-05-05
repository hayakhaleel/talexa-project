from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class SessionPaths:
    session_number: int
    session_dir: Path
    input_dir: Path
    intermediate_dir: Path
    output_dir: Path
    stored_pdf_path: Path
    stored_audio_path: Path | None
    stored_portrait_path: Path | None

    @property
    def stored_input_path(self) -> Path:
        """Backward-compatible alias for the primary PDF input."""
        return self.stored_pdf_path


def _next_session_number(sessions_root: Path) -> int:
    existing_numbers = []

    for child in sessions_root.iterdir():
        if child.is_dir() and child.name.isdigit():
            existing_numbers.append(int(child.name))

    return max(existing_numbers, default=0) + 1


def get_next_session_number(project_root: Path) -> int:
    sessions_root = project_root / "sessions"
    sessions_root.mkdir(parents=True, exist_ok=True)
    return _next_session_number(sessions_root)


def _resolve_required_file(file_path: str, label: str) -> Path:
    resolved_path = Path(file_path).expanduser().resolve()
    if not resolved_path.exists():
        raise FileNotFoundError(f"{label} file not found: {resolved_path}")

    return resolved_path


def _copy_optional_file(source_path: str | None, destination_dir: Path) -> Path | None:
    if not source_path:
        return None

    resolved_path = Path(source_path).expanduser().resolve()
    if not resolved_path.exists():
        raise FileNotFoundError(f"Input file not found: {resolved_path}")

    stored_path = destination_dir / resolved_path.name
    shutil.copy2(resolved_path, stored_path)
    return stored_path


def create_session(
    project_root: Path,
    pdf_file_path: str,
    audio_file_path: str | None = None,
    portrait_file_path: str | None = None,
    session_number: int | None = None,
) -> SessionPaths:
    pdf_path = _resolve_required_file(pdf_file_path, "PDF")

    sessions_root = project_root / "sessions"
    sessions_root.mkdir(parents=True, exist_ok=True)

    if session_number is None:
        session_number = _next_session_number(sessions_root)
    session_dir = sessions_root / str(session_number)
    input_dir = session_dir / "input"
    intermediate_dir = session_dir / "intermediate"
    output_dir = session_dir / "output"

    input_dir.mkdir(parents=True, exist_ok=True)
    intermediate_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    stored_pdf_path = input_dir / pdf_path.name
    shutil.copy2(pdf_path, stored_pdf_path)

    stored_audio_path = _copy_optional_file(audio_file_path, input_dir)
    stored_portrait_path = _copy_optional_file(portrait_file_path, input_dir)

    return SessionPaths(
        session_number=session_number,
        session_dir=session_dir,
        input_dir=input_dir,
        intermediate_dir=intermediate_dir,
        output_dir=output_dir,
        stored_pdf_path=stored_pdf_path,
        stored_audio_path=stored_audio_path,
        stored_portrait_path=stored_portrait_path,
    )
