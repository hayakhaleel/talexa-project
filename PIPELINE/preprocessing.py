from __future__ import annotations

import tempfile
from pathlib import Path

import fitz
import numpy as np
import soundfile as sf
from PIL import Image

from PIPELINE.portrait_classifier import get_portrait_classifier


MAX_TEXTBOOK_PAGES = 20
MAX_SLIDES_PAGES = 50
REQUIRED_AUDIO_SECONDS = 6.0
KEEP_AUDIO_SEGMENT = "start"
QUIET_AUDIO_RMS_THRESHOLD = 0.08
TARGET_AUDIO_PEAK = 0.85


def _ask_required_path(label: str) -> str:
    path_value = input(f"Enter the full path of the {label}: ").strip().strip('"')
    if not path_value:
        raise ValueError(f"A {label} path is required.")
    return path_value


def _resolve_existing_file(file_path: str, label: str) -> Path:
    resolved_path = Path(file_path).expanduser().resolve()
    if not resolved_path.exists() or not resolved_path.is_file():
        raise FileNotFoundError(f"{label} file not found: {resolved_path}")
    return resolved_path


def _count_pdf_pages(pdf_path: Path) -> int:
    with fitz.open(pdf_path) as document:
        return len(document)


def _validate_source_document(source_path: str, source_type: str) -> Path:
    resolved_path = _resolve_existing_file(source_path, "Source document")

    if resolved_path.suffix.lower() != ".pdf":
        raise ValueError("The source document must be a PDF file.")

    page_count = _count_pdf_pages(resolved_path)
    max_pages = MAX_TEXTBOOK_PAGES if source_type == "textbook" else MAX_SLIDES_PAGES
    unit_name = "pages" if source_type == "textbook" else "slides"

    if page_count > max_pages:
        raise ValueError(
            f"The {source_type} has {page_count} {unit_name}. "
            f"The limit is {max_pages}."
        )

    print(f"{source_type.title()} accepted with {page_count} {unit_name}.")
    return resolved_path


def validate_source_document(source_path: str, source_type: str) -> Path:
    return _validate_source_document(source_path, source_type)


def _read_audio_mono(audio_path: Path) -> tuple[np.ndarray, int]:
    audio, sample_rate = sf.read(audio_path, dtype="float32")
    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)
    return audio, sample_rate


def _raise_audio_volume_if_needed(audio: np.ndarray) -> np.ndarray:
    if audio.size == 0:
        return audio

    peak = float(np.max(np.abs(audio)))
    rms = float(np.sqrt(np.mean(np.square(audio))))

    if rms >= QUIET_AUDIO_RMS_THRESHOLD or peak <= 0.0:
        return audio

    gain = TARGET_AUDIO_PEAK / peak
    boosted_audio = audio * gain
    return np.clip(boosted_audio, -0.99, 0.99)


def _trim_audio_to_required_length(audio: np.ndarray, sample_rate: int) -> np.ndarray:
    max_samples = int(REQUIRED_AUDIO_SECONDS * sample_rate)
    if KEEP_AUDIO_SEGMENT == "end":
        return audio[-max_samples:]
    return audio[:max_samples]


def _store_processed_audio(audio: np.ndarray, sample_rate: int, source_path: Path) -> Path:
    output_dir = Path(tempfile.gettempdir()) / "talexa_preprocessing"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{source_path.stem}_preprocessed.wav"
    sf.write(output_path, audio, sample_rate)
    return output_path


def _ensure_png_portrait(image_path: Path) -> Path:
    with Image.open(image_path) as image:
        if image.format == "PNG":
            return image_path

        output_dir = Path(tempfile.gettempdir()) / "talexa_preprocessing"
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"{image_path.stem}_portrait.png"

        converted_image = image.convert("RGBA") if "A" in image.getbands() else image.convert("RGB")
        converted_image.save(output_path, format="PNG")

    print(f"Portrait image was converted to PNG: {output_path}")
    return output_path


def _validate_and_prepare_audio(audio_path: str) -> Path:
    resolved_path = _resolve_existing_file(audio_path, "Audio")
    audio, sample_rate = _read_audio_mono(resolved_path)

    if audio.size == 0:
        raise ValueError("The audio file is empty.")

    duration_seconds = len(audio) / sample_rate
    if duration_seconds < REQUIRED_AUDIO_SECONDS:
        raise ValueError(
            f"The audio is {duration_seconds:.2f} seconds long. "
            f"Please enter an audio sample that is at least 6 seconds."
        )

    processed_audio = audio.copy()
    if duration_seconds > REQUIRED_AUDIO_SECONDS:
        processed_audio = _trim_audio_to_required_length(processed_audio, sample_rate)
        if KEEP_AUDIO_SEGMENT == "end":
            print("Audio was longer than 6 seconds, so the beginning was cut.")
        else:
            print("Audio was longer than 6 seconds, so the end was cut.")

    boosted_audio = _raise_audio_volume_if_needed(processed_audio)
    if not np.array_equal(boosted_audio, processed_audio):
        print("Audio volume was low, so it was increased for clarity.")
    processed_audio = boosted_audio

    processed_path = _store_processed_audio(processed_audio, sample_rate, resolved_path)
    print(f"Audio accepted and saved to: {processed_path}")
    return processed_path


def validate_and_prepare_audio(audio_path: str) -> Path:
    return _validate_and_prepare_audio(audio_path)


def _validate_portrait_image(image_path: str) -> Path:
    resolved_path = _resolve_existing_file(image_path, "Portrait image")
    png_path = _ensure_png_portrait(resolved_path)
    classifier = get_portrait_classifier()
    classification = classifier.classify(png_path)

    if not classification["is_valid"]:
        raise ValueError(
            "The portrait image was classified as invalid "
            f"(label={classification['label']}, confidence={classification['confidence']:.3f})."
        )

    print(
        "Portrait image accepted by classifier "
        f"(confidence={classification['confidence']:.3f})."
    )
    return png_path


def validate_portrait_image(image_path: str) -> Path:
    return _validate_portrait_image(image_path)


def preprocess_inputs(source_type: str) -> tuple[str, str, str]:
    source_path: Path | None = None
    audio_path: Path | None = None
    portrait_path: Path | None = None

    while source_path is None or audio_path is None or portrait_path is None:
        if source_path is None:
            try:
                entered_source_path = _ask_required_path(
                    "textbook PDF" if source_type == "textbook" else "slides PDF"
                )
                source_path = _validate_source_document(entered_source_path, source_type)
            except Exception as exc:
                print(f"Invalid source document: {exc}")

        if audio_path is None:
            try:
                entered_audio_path = _ask_required_path("reference audio")
                audio_path = _validate_and_prepare_audio(entered_audio_path)
            except Exception as exc:
                print(f"Invalid audio sample: {exc}")

        if portrait_path is None:
            try:
                entered_portrait_path = _ask_required_path("portrait image")
                portrait_path = _validate_portrait_image(entered_portrait_path)
            except Exception as exc:
                print(f"Invalid portrait image: {exc}")

    return str(source_path), str(audio_path), str(portrait_path)
