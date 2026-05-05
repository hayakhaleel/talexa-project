from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont


PROJECT_ROOT = Path(__file__).resolve().parents[1]

SLIDE_DIR_PREFIX = "SLIDES_"
ENGLISH_SUBTITLE_PREFIX = "SUBTITLE_"
CURSOR_PREFIX = "CURSOR_"
TALKING_HEAD_PREFIX = "TALKING_HEAD"
ARABIC_SUBTITLE_HINTS = ("arabic", "translated", "translation")
CURSOR_IMAGE_PATH = PROJECT_ROOT / "PIPELINE" / "cursor.png"
CURSOR_HOTSPOT_X = 0.12
CURSOR_HOTSPOT_Y = 0.08
TALKING_HEAD_WIDTH_PX = 300
SLIDE_TRANSITION_HOLD_SECONDS = 1.0


@dataclass(frozen=True)
class AssemblyInputs:
    session_dir: Path
    intermediate_dir: Path
    output_dir: Path
    slides_dir: Path
    subtitle_path: Path
    cursor_path: Path
    talking_head_video_path: Path
@dataclass(frozen=True)
class SubtitleSegment:
    text: str
    start: float
    end: float
    focus: str


@dataclass(frozen=True)
class SlideSegment:
    slide_number: int
    image_path: Path
    start: float
    end: float
    subtitles: list[SubtitleSegment]


def _normalize_language(language: str) -> str:
    normalized = language.strip().lower()
    if normalized not in {"english", "arabic"}:
        raise ValueError("language must be either 'english' or 'arabic'.")
    return normalized


def _resolve_session_dir(session_ref: str | int) -> Path:
    session_dir = PROJECT_ROOT / "sessions" / str(session_ref)
    if not session_dir.exists():
        raise FileNotFoundError(f"Session directory not found: {session_dir}")
    return session_dir


def _require_single_match(matches: list[Path], label: str, search_dir: Path) -> Path:
    if not matches:
        raise FileNotFoundError(f"Could not find {label} in: {search_dir}")
    if len(matches) > 1:
        joined = ", ".join(path.name for path in matches)
        raise FileExistsError(f"Expected one {label} in {search_dir}, found: {joined}")
    return matches[0]


def _find_media_dir(intermediate_dir: Path, preferred_names: tuple[str, ...], patterns: tuple[str, ...]) -> Path:
    for name in preferred_names:
        candidate = intermediate_dir / name
        if candidate.is_dir() and any(candidate.glob(patterns[0])):
            return candidate
        if candidate.is_dir() and any(candidate.glob(patterns[1])):
            return candidate

    for candidate in sorted(path for path in intermediate_dir.iterdir() if path.is_dir()):
        if any(candidate.glob(patterns[0])) or any(candidate.glob(patterns[1])):
            return candidate

    raise FileNotFoundError(f"Could not find media directory in: {intermediate_dir}")


def _discover_talking_head_video(intermediate_dir: Path) -> Path:
    preferred_dirs = sorted(
        path
        for path in intermediate_dir.iterdir()
        if path.is_dir() and path.name.upper().startswith(TALKING_HEAD_PREFIX)
    )

    for directory in preferred_dirs:
        direct_matches = sorted(directory.glob("*.mp4"))
        if direct_matches:
            return direct_matches[0]
        recursive_matches = sorted(directory.rglob("*.mp4"))
        if recursive_matches:
            return recursive_matches[0]

    generic_matches = sorted(
        path
        for path in intermediate_dir.rglob("*.mp4")
        if "assembly" not in path.name.lower()
    )
    if generic_matches:
        return generic_matches[0]

    raise FileNotFoundError(
        f"Could not find a talking-head mp4 in the intermediate folder: {intermediate_dir}"
    )


def discover_inputs(session_ref: str | int, language: str) -> AssemblyInputs:
    normalized_language = _normalize_language(language)
    session_dir = _resolve_session_dir(session_ref)
    intermediate_dir = session_dir / "intermediate"
    output_dir = session_dir / "output"

    try:
        slides_dir = _require_single_match(
            sorted(path for path in intermediate_dir.iterdir() if path.is_dir() and path.name.startswith(SLIDE_DIR_PREFIX)),
            "slides image directory",
            intermediate_dir,
        )
    except FileNotFoundError:
        slides_dir = _find_media_dir(intermediate_dir, ("slide_images",), ("*.png", "*.jpg"))

    cursor_candidates = sorted(path for path in intermediate_dir.glob(f"{CURSOR_PREFIX}*.json"))
    if not cursor_candidates:
        cursor_candidates = sorted(path for path in intermediate_dir.glob("cursor*.json"))
    cursor_path = _require_single_match(cursor_candidates, "cursor json", intermediate_dir)
    talking_head_video_path = _discover_talking_head_video(intermediate_dir)

    if normalized_language == "english":
        subtitle_candidates = sorted(
            path
            for path in intermediate_dir.glob(f"{ENGLISH_SUBTITLE_PREFIX}*.json")
            if not any(hint in path.stem.lower() for hint in ARABIC_SUBTITLE_HINTS)
        )
        if not subtitle_candidates:
            subtitle_candidates = sorted(path for path in intermediate_dir.glob("subtitles*.json"))
    else:
        subtitle_candidates = sorted(
            path
            for path in intermediate_dir.glob("*.json")
            if any(hint in path.stem.lower() for hint in ARABIC_SUBTITLE_HINTS)
        )
        if not subtitle_candidates:
            raise FileNotFoundError(
                "Arabic subtitle JSON was not found. Add a translated subtitle file to the session "
                f"intermediate folder: {intermediate_dir}"
            )

    subtitle_path = _require_single_match(subtitle_candidates, f"{normalized_language} subtitle json", intermediate_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    return AssemblyInputs(
        session_dir=session_dir,
        intermediate_dir=intermediate_dir,
        output_dir=output_dir,
        slides_dir=slides_dir,
        subtitle_path=subtitle_path,
        cursor_path=cursor_path,
        talking_head_video_path=talking_head_video_path,
    )


def _load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _resolve_slide_image(slides_dir: Path, slide_number: int, declared_image: str | None) -> Path:
    if declared_image:
        declared_name = Path(declared_image).name
        candidate = slides_dir / declared_name
        if candidate.exists():
            return candidate

    candidates = sorted(slides_dir.glob(f"*{slide_number:03d}*.png"))
    if not candidates:
        candidates = sorted(slides_dir.glob(f"*_{slide_number}.png"))
    if not candidates:
        raise FileNotFoundError(f"Could not resolve image for slide {slide_number} in {slides_dir}")
    return candidates[0]


def build_slide_segments(inputs: AssemblyInputs) -> list[SlideSegment]:
    subtitle_data = _load_json(inputs.subtitle_path)
    cursor_data = _load_json(inputs.cursor_path)

    if not isinstance(subtitle_data, dict):
        raise ValueError(f"Unsupported subtitle format in {inputs.subtitle_path}")
    if not isinstance(cursor_data, list):
        raise ValueError(f"Unsupported cursor format in {inputs.cursor_path}")

    slide_entries = sorted(
        subtitle_data.values(),
        key=lambda entry: int(entry.get("slide_number", 0)),
    )

    segments: list[SlideSegment] = []
    cursor_index = 0

    for slide_entry in slide_entries:
        slide_number = int(slide_entry["slide_number"])
        items = slide_entry.get("items", [])
        if not items:
            continue

        if cursor_index + len(items) > len(cursor_data):
            raise ValueError(
                f"Not enough cursor segments for slide {slide_number}. "
                f"Expected {len(items)} more entries, found {len(cursor_data) - cursor_index}."
            )

        image_path = _resolve_slide_image(inputs.slides_dir, slide_number, slide_entry.get("image"))
        per_slide_segments: list[SubtitleSegment] = []

        for item in items:
            cursor_segment = cursor_data[cursor_index]
            cursor_index += 1
            per_slide_segments.append(
                SubtitleSegment(
                    text=str(item.get("sentence", "")).strip(),
                    start=float(cursor_segment["start"]),
                    end=float(cursor_segment["end"]),
                    focus=str(cursor_segment.get("focus", item.get("focus", ""))),
                )
            )

        start_time = per_slide_segments[0].start
        end_time = per_slide_segments[-1].end

        segments.append(
            SlideSegment(
                slide_number=slide_number,
                image_path=image_path,
                start=start_time,
                end=end_time,
                subtitles=per_slide_segments,
            )
        )

    if cursor_index < len(cursor_data):
        raise ValueError(
            f"Cursor file has {len(cursor_data) - cursor_index} unused entries. "
            "The current implementation expects one cursor segment per subtitle item."
        )

    if not segments:
        raise ValueError(f"No slide segments could be built from: {inputs.subtitle_path}")

    return segments


def _find_font_path() -> Path | None:
    candidate_paths = [
        Path(os.environ.get("WINDIR", r"C:\Windows")) / "Fonts" / "arial.ttf",
        Path(os.environ.get("WINDIR", r"C:\Windows")) / "Fonts" / "tahoma.ttf",
        Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
        Path("/Library/Fonts/Arial.ttf"),
    ]
    for candidate in candidate_paths:
        if candidate.exists():
            return candidate
    return None


def _load_font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    font_path = _find_font_path()
    if font_path is not None:
        return ImageFont.truetype(str(font_path), size=size)
    return ImageFont.load_default()


def _wrap_text(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.ImageFont, max_width: int) -> list[str]:
    words = text.split()
    if not words:
        return [""]

    lines: list[str] = []
    current = words[0]

    for word in words[1:]:
        trial = f"{current} {word}"
        bbox = draw.textbbox((0, 0), trial, font=font)
        if bbox[2] - bbox[0] <= max_width:
            current = trial
        else:
            lines.append(current)
            current = word

    lines.append(current)
    return lines


def _current_subtitle(slide: SlideSegment, timestamp: float) -> SubtitleSegment | None:
    for segment in slide.subtitles:
        if segment.start <= timestamp < segment.end:
            return segment
    if slide.subtitles and slide.subtitles[-1].end <= timestamp <= slide.end:
        return slide.subtitles[-1]
    if math.isclose(timestamp, slide.end, abs_tol=1e-3):
        return slide.subtitles[-1]
    return None


def _cursor_for_time(cursor_segments: list[dict[str, Any]], timestamp: float) -> tuple[float, float] | None:
    for segment in cursor_segments:
        if float(segment["start"]) <= timestamp < float(segment["end"]):
            cursor = segment.get("cursor")
            if isinstance(cursor, list) and len(cursor) == 2:
                return float(cursor[0]), float(cursor[1])
            return None
    if cursor_segments and math.isclose(timestamp, float(cursor_segments[-1]["end"]), abs_tol=1e-3):
        cursor = cursor_segments[-1].get("cursor")
        if isinstance(cursor, list) and len(cursor) == 2:
            return float(cursor[0]), float(cursor[1])
    return None


def _load_cursor_sprite(scale_x: float, scale_y: float) -> Image.Image | None:
    if not CURSOR_IMAGE_PATH.exists():
        return None

    with Image.open(CURSOR_IMAGE_PATH) as cursor_image:
        cursor_rgba = cursor_image.convert("RGBA")

    target_height = max(20, int(36 * ((scale_x + scale_y) / 2)))
    aspect_ratio = cursor_rgba.width / max(cursor_rgba.height, 1)
    target_width = max(12, int(target_height * aspect_ratio))
    return cursor_rgba.resize((target_width, target_height), Image.Resampling.LANCZOS)


def _paste_cursor(
    overlay: Image.Image,
    point: tuple[float, float],
    scale_x: float,
    scale_y: float,
) -> None:
    cursor_sprite = _load_cursor_sprite(scale_x, scale_y)
    if cursor_sprite is None:
        return

    x = int(point[0] * scale_x)
    y = int(point[1] * scale_y)

    hotspot_x = int(round(cursor_sprite.width * CURSOR_HOTSPOT_X))
    hotspot_y = int(round(cursor_sprite.height * CURSOR_HOTSPOT_Y))

    top_left_x = x - hotspot_x
    top_left_y = y - hotspot_y

    # Keep the cursor fully inside the frame while preserving the tip anchor.
    top_left_x = max(min(top_left_x, overlay.width - cursor_sprite.width), -cursor_sprite.width + 1)
    top_left_y = max(min(top_left_y, overlay.height - cursor_sprite.height), -cursor_sprite.height + 1)

    overlay.alpha_composite(cursor_sprite, (top_left_x, top_left_y))


def _render_frame(
    slide: SlideSegment,
    timestamp: float,
    cursor_segments: list[dict[str, Any]],
    base_slide_image: Image.Image,
    original_slide_size: tuple[int, int],
    font: ImageFont.ImageFont,
) -> np.ndarray:
    frame_width, frame_height = base_slide_image.size
    slide_image = base_slide_image.copy()
    overlay = Image.new("RGBA", (frame_width, frame_height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    subtitle = _current_subtitle(slide, timestamp)
    if subtitle:
        text_margin = int(frame_width * 0.07)
        max_text_width = frame_width - (text_margin * 2)
        text_lines = _wrap_text(draw, subtitle.text, font, max_text_width)
        line_height = font.getbbox("Ag")[3] - font.getbbox("Ag")[1] + 6
        text_block_height = line_height * len(text_lines)
        text_y = frame_height - text_block_height - 42
        for line in text_lines:
            bbox = draw.textbbox((0, 0), line, font=font)
            line_width = bbox[2] - bbox[0]
            text_x = (frame_width - line_width) / 2
            draw.text((text_x, text_y), line, font=font, fill=(0, 0, 0, 255))
            text_y += line_height

    raw_cursor = _cursor_for_time(cursor_segments, timestamp)
    if raw_cursor:
        scale_x = frame_width / max(original_slide_size[0], 1)
        scale_y = frame_height / max(original_slide_size[1], 1)
        _paste_cursor(overlay, raw_cursor, scale_x, scale_y)

    composed = Image.alpha_composite(slide_image.convert("RGBA"), overlay)
    return cv2.cvtColor(np.array(composed.convert("RGB")), cv2.COLOR_RGB2BGR)


def _resolve_ffmpeg(ffmpeg_path: str | None) -> str | None:
    if ffmpeg_path:
        candidate = Path(ffmpeg_path).expanduser()
        if not candidate.exists():
            raise FileNotFoundError(f"ffmpeg executable not found: {candidate}")
        return str(candidate)

    for env_var in ("FFMPEG_PATH", "IMAGEIO_FFMPEG_EXE"):
        value = os.environ.get(env_var)
        if value:
            candidate = Path(value).expanduser()
            if candidate.exists():
                return str(candidate)

    return shutil.which("ffmpeg")


def overlay_talking_head_video(
    base_video_path: Path,
    talking_head_video_path: Path,
    output_path: Path,
    ffmpeg_path: str | None,
) -> bool:
    ffmpeg_executable = _resolve_ffmpeg(ffmpeg_path)
    if ffmpeg_executable is None:
        return False

    filter_complex = (
        f"[1:v][0:v]scale2ref=w='min({TALKING_HEAD_WIDTH_PX},main_w)':h=ow/mdar[th][base];"
        "[base][th]overlay=main_w-overlay_w:0[v]"
    )

    command = [
        ffmpeg_executable,
        "-y",
        "-i",
        str(base_video_path),
        "-i",
        str(talking_head_video_path),
        "-filter_complex",
        filter_complex,
        "-map",
        "[v]",
        "-map",
        "1:a:0",
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-crf",
        "18",
        "-c:a",
        "aac",
        "-shortest",
        str(output_path),
    ]
    subprocess.run(command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    subprocess.run(command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    return True


def build_silent_video_from_segment_frames(
    slides: list[SlideSegment],
    cursor_segments: list[dict[str, Any]],
    output_path: Path,
    ffmpeg_path: str | None,
    fps: int,
) -> bool:
    ffmpeg_executable = _resolve_ffmpeg(ffmpeg_path)
    if ffmpeg_executable is None:
        return False

    with Image.open(slides[0].image_path) as first_image:
        frame_size = (first_image.width, first_image.height)
    font_size = max(16, int(frame_size[1] * 0.022))
    font = _load_font(font_size)

    temp_dir = output_path.parent
    concat_path = temp_dir / "segments.txt"
    segment_files: list[tuple[Path, float]] = []
    segment_index = 0
    frame_duration = 1 / max(fps, 1)

    for slide in slides:
        with Image.open(slide.image_path) as original_slide:
            original_size = original_slide.size
            base_slide_image = original_slide.convert("RGB").resize(frame_size, Image.Resampling.LANCZOS)

            for subtitle_index, subtitle_segment in enumerate(slide.subtitles):
                raw_duration = max(subtitle_segment.end - subtitle_segment.start, frame_duration)
                frame_count = max(1, math.ceil(raw_duration * fps))
                if subtitle_index == len(slide.subtitles) - 1:
                    extra_hold_frames = math.ceil(SLIDE_TRANSITION_HOLD_SECONDS * fps)
                    frame_count += max(1, extra_hold_frames)
                duration = frame_count / fps
                timestamp = min(subtitle_segment.start + 1e-4, subtitle_segment.end)
                frame = _render_frame(
                    slide=slide,
                    timestamp=timestamp,
                    cursor_segments=cursor_segments,
                    base_slide_image=base_slide_image,
                    original_slide_size=original_size,
                    font=font,
                )
                frame_path = temp_dir / f"segment_{segment_index:05d}.png"
                cv2.imwrite(str(frame_path), frame)
                segment_files.append((frame_path, duration))
                segment_index += 1

    with concat_path.open("w", encoding="utf-8") as handle:
        for frame_path, duration in segment_files:
            escaped = frame_path.as_posix().replace("'", "'\\''")
            handle.write(f"file '{escaped}'\n")
            handle.write(f"duration {duration:.6f}\n")
        if segment_files:
            escaped = segment_files[-1][0].as_posix().replace("'", "'\\''")
            handle.write(f"file '{escaped}'\n")

    command = [
        ffmpeg_executable,
        "-y",
        "-f",
        "concat",
        "-safe",
        "0",
        "-i",
        str(concat_path),
        "-vf",
        f"fps={fps},format=yuv420p",
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-crf",
        "18",
        str(output_path),
    ]
    subprocess.run(command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    return True


def assemble_session_video(
    *,
    session_id: str | int,
    language: str,
    fps: int = 24,
    ffmpeg_path: str | None = None,
) -> dict[str, str]:
    inputs = discover_inputs(session_id, language)
    slides = build_slide_segments(inputs)
    cursor_segments = _load_json(inputs.cursor_path)

    session_label = str(session_id)
    safe_session_label = session_label.replace("\\", "_").replace("/", "_")
    temp_dir = Path(tempfile.mkdtemp(prefix=f"talexa_assembly_{safe_session_label}_"))
    silent_video_path = temp_dir / f"session_{safe_session_label}_{language}_silent.mp4"
    final_output_path = inputs.output_dir / f"ASSEMBLY_{language}_{safe_session_label}.mp4"

    result: dict[str, str] = {
        "silent_video_path": str(silent_video_path),
        "video_output_path": str(final_output_path),
    }
    try:
        if not build_silent_video_from_segment_frames(
            slides=slides,
            cursor_segments=cursor_segments,
            output_path=silent_video_path,
            ffmpeg_path=ffmpeg_path,
            fps=fps,
        ):
            raise RuntimeError("ffmpeg is required to build the optimized assembly video.")

        if overlay_talking_head_video(
            silent_video_path,
            inputs.talking_head_video_path,
            final_output_path,
            ffmpeg_path=ffmpeg_path,
        ):
            return result

        shutil.copy2(silent_video_path, final_output_path)
        result["warning"] = (
            "ffmpeg was not found, so the final MP4 was created without audio muxed from the talking-head video."
        )
        return result
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Assemble a lecture video from a session's intermediate assets.")
    parser.add_argument("--session-id", required=True, help="Session folder name under the sessions directory.")
    parser.add_argument("--language", choices=["english", "arabic"], required=True, help="Subtitle language to use.")
    parser.add_argument("--fps", type=int, default=24, help="Frames per second for the output video.")
    parser.add_argument("--ffmpeg-path", default=None, help="Optional explicit path to ffmpeg for audio muxing.")
    args = parser.parse_args()

    result = assemble_session_video(
        session_id=args.session_id,
        language=args.language,
        fps=args.fps,
        ffmpeg_path=args.ffmpeg_path,
    )

    print(f"Video output: {result['video_output_path']}")
    if "audio_track_path" in result:
        print(f"Audio track: {result['audio_track_path']}")
    if "warning" in result:
        print(f"Warning: {result['warning']}")


if __name__ == "__main__":
    main()
