from __future__ import annotations

import json
import os
import re
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np


@dataclass(frozen=True)
class TextRegion:
    x: int
    y: int
    w: int
    h: int

    @property
    def center(self) -> tuple[float, float]:
        return (self.x + self.w / 2, self.y + self.h / 2)

    @property
    def area(self) -> int:
        return self.w * self.h


@dataclass(frozen=True)
class FrameElement:
    kind: str
    text: str


class CursorAgent:
    """
    Fast layout-based cursor agent.

    Instead of asking a vision model to ground every focus prompt, this agent:
    1. detects text regions on the slide image using OpenCV,
    2. separates title-like regions from body regions,
    3. maps each subtitle focus to a likely region based on focus type and item order,
    4. emits the same cursor timeline JSON format expected by the pipeline.
    """

    def __init__(
        self,
        model: str = "layout-fast",
        images_dir: str = "Data/intermediate/slide_images",
        audio_dir: str = "Data/intermediate/slide_audios",
        slides_tex_path: str | None = None,
    ) -> None:
        self.model = model
        self.images_dir = images_dir
        self.audio_dir = audio_dir
        self.slides_tex_path = slides_tex_path
        self.frame_elements_by_slide = self._load_frame_elements(slides_tex_path)

    def _extract_index(self, name: str) -> int:
        matches = re.findall(r"\d+", name)
        return int(matches[-1]) if matches else 0

    def _sorted_media_files(self, folder: str, suffixes: tuple[str, ...]) -> list[Path]:
        folder_path = Path(folder)
        if not folder_path.exists():
            raise FileNotFoundError(f"Folder not found: {folder_path}")

        files = [
            path
            for path in folder_path.iterdir()
            if path.is_file() and path.suffix.lower() in suffixes
        ]
        return sorted(files, key=lambda path: self._extract_index(path.name))

    def _get_audio_duration(self, audio_path: Path) -> float:
        with wave.open(str(audio_path), "rb") as wav_file:
            frame_rate = wav_file.getframerate()
            frame_count = wav_file.getnframes()
            if frame_rate <= 0:
                raise ValueError(f"Invalid WAV frame rate for: {audio_path}")
            return frame_count / frame_rate

    def _load_image(self, image_path: Path) -> np.ndarray:
        image = cv2.imread(str(image_path))
        if image is None:
            raise FileNotFoundError(f"Could not read slide image: {image_path}")
        return image

    def _normalize_text(self, text: str) -> str:
        text = text.lower()
        text = re.sub(r"\\textbf\{([^}]*)\}", r"\1", text)
        text = re.sub(r"\\[a-zA-Z]+\*?(?:\[[^\]]*\])?\{([^}]*)\}", r"\1", text)
        text = re.sub(r"[^a-z0-9\s]+", " ", text)
        text = re.sub(r"\s+", " ", text).strip()
        return text

    def _strip_tex_markup(self, text: str) -> str:
        text = text.replace("~", " ")
        text = text.replace("\\&", "&")
        text = text.replace("\\%", "%")
        text = text.replace("\\_", "_")
        text = re.sub(r"\\textbf\{([^}]*)\}", r"\1", text)
        text = re.sub(r"\\emph\{([^}]*)\}", r"\1", text)
        text = re.sub(r"\\[a-zA-Z]+\*?(?:\[[^\]]*\])?\{([^}]*)\}", r"\1", text)
        text = re.sub(r"\\[a-zA-Z]+\*?", " ", text)
        text = text.replace("{", " ").replace("}", " ")
        text = re.sub(r"\s+", " ", text).strip()
        return text

    def _extract_quoted_strings(self, text: str) -> list[str]:
        return [match.strip() for match in re.findall(r"'([^']+)'|\"([^\"]+)\"", text) for match in match if match.strip()]

    def _extract_frame_bodies(self, tex_text: str) -> list[str]:
        frame_blocks: list[str] = []
        pattern = re.compile(r"\\begin\{frame\}(?:\[[^\]]*\])?(?:\{([^}]*)\})?(.*?)\\end\{frame\}", re.DOTALL)
        for title, body in pattern.findall(tex_text):
            title_text = self._strip_tex_markup(title)
            block = body
            if title_text:
                block = f"FRAME_TITLE::{title_text}\n{body}"
            frame_blocks.append(block)
        return frame_blocks

    def _parse_frame_elements(self, frame_body: str) -> list[FrameElement]:
        elements: list[FrameElement] = []

        title_match = re.search(r"FRAME_TITLE::([^\n]+)", frame_body)
        if title_match:
            elements.append(FrameElement(kind="frame_title", text=title_match.group(1).strip()))

        block_pattern = re.compile(r"\\begin\{block\}\{([^}]*)\}(.*?)\\end\{block\}", re.DOTALL)
        for block_title, block_content in block_pattern.findall(frame_body):
            clean_title = self._strip_tex_markup(block_title)
            if clean_title:
                elements.append(FrameElement(kind="block_title", text=clean_title))

            clean_content = self._strip_tex_markup(block_content)
            for part in re.split(r"\s*-\s+", clean_content):
                part = part.strip()
                if not part:
                    continue
                kind = "block_body"
                if any(token in part.lower() for token in ("depth-first", "breadth-first", "heuristic", "supervised", "unsupervised")):
                    kind = "bullet"
                elements.append(FrameElement(kind=kind, text=part))

        item_pattern = re.compile(r"\\item\s+(.*?)(?=\\item|\\end\{itemize\})", re.DOTALL)
        for item_text in item_pattern.findall(frame_body):
            clean_item = self._strip_tex_markup(item_text)
            if clean_item:
                elements.append(FrameElement(kind="bullet", text=clean_item))

        body_without_blocks = re.sub(r"\\begin\{block\}.*?\\end\{block\}", " ", frame_body, flags=re.DOTALL)
        body_without_items = re.sub(r"\\begin\{itemize\}.*?\\end\{itemize\}", " ", body_without_blocks, flags=re.DOTALL)
        clean_body = self._strip_tex_markup(body_without_items)
        for sentence in re.split(r"(?<=[.!?])\s+", clean_body):
            sentence = sentence.strip(" -")
            if sentence and not sentence.startswith("FRAME_TITLE::"):
                elements.append(FrameElement(kind="paragraph", text=sentence))

        unique: list[FrameElement] = []
        seen: set[tuple[str, str]] = set()
        for element in elements:
            key = (element.kind, self._normalize_text(element.text))
            if key[1] and key not in seen:
                seen.add(key)
                unique.append(element)
        return unique

    def _load_frame_elements(self, slides_tex_path: str | None) -> dict[int, list[FrameElement]]:
        if not slides_tex_path:
            return {}

        path = Path(slides_tex_path)
        if not path.exists():
            return {}

        tex_text = path.read_text(encoding="utf-8")
        frames = self._extract_frame_bodies(tex_text)
        mapping: dict[int, list[FrameElement]] = {}
        for slide_number, frame_body in enumerate(frames, start=1):
            mapping[slide_number] = self._parse_frame_elements(frame_body)
        return mapping

    def _token_overlap_score(self, left: str, right: str) -> float:
        left_tokens = set(self._normalize_text(left).split())
        right_tokens = set(self._normalize_text(right).split())
        if not left_tokens or not right_tokens:
            return 0.0
        overlap = len(left_tokens & right_tokens)
        if overlap == 0:
            return 0.0
        return overlap / max(min(len(left_tokens), len(right_tokens)), 1)

    def _detect_text_regions(self, image: np.ndarray) -> list[TextRegion]:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Adaptive threshold works better across light/dark slide themes.
        binary = cv2.adaptiveThreshold(
            gray,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            31,
            11,
        )

        # Merge nearby letters into line-like blobs.
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 7))
        merged = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=1)

        contours, _ = cv2.findContours(merged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        height, width = gray.shape
        min_area = max(150, int(width * height * 0.00025))

        regions: list[TextRegion] = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            area = w * h
            if area < min_area:
                continue
            if w < 20 or h < 8:
                continue
            if w > width * 0.98 and h > height * 0.85:
                continue
            if y + h >= height * 0.93:
                continue
            regions.append(TextRegion(x=x, y=y, w=w, h=h))

        regions.sort(key=lambda region: (region.y, region.x))
        return self._merge_regions_on_same_line(regions)

    def _merge_regions_on_same_line(self, regions: list[TextRegion]) -> list[TextRegion]:
        if not regions:
            return []

        merged: list[TextRegion] = []
        current = regions[0]
        for region in regions[1:]:
            same_line = abs(region.y - current.y) <= max(current.h, region.h) * 0.6
            close_horizontally = region.x <= current.x + current.w + 30
            if same_line and close_horizontally:
                x1 = min(current.x, region.x)
                y1 = min(current.y, region.y)
                x2 = max(current.x + current.w, region.x + region.w)
                y2 = max(current.y + current.h, region.y + region.h)
                current = TextRegion(x=x1, y=y1, w=x2 - x1, h=y2 - y1)
            else:
                merged.append(current)
                current = region
        merged.append(current)
        return merged

    def _split_title_and_body_regions(
        self,
        regions: list[TextRegion],
        image_height: int,
    ) -> tuple[list[TextRegion], list[TextRegion]]:
        if not regions:
            return [], []

        title_band = image_height * 0.28
        title_regions = [region for region in regions if region.y + region.h / 2 <= title_band]
        body_regions = [region for region in regions if region not in title_regions]

        if not title_regions and regions:
            title_regions = [min(regions, key=lambda region: region.y)]
            body_regions = [region for region in regions if region not in title_regions]

        return title_regions, body_regions

    def _choose_title_region(self, title_regions: list[TextRegion], fallback: TextRegion) -> TextRegion:
        if not title_regions:
            return fallback
        return max(title_regions, key=lambda region: (region.area, -region.y))

    def _body_region_sequence(self, body_regions: list[TextRegion], fallback: TextRegion) -> list[TextRegion]:
        filtered_regions = [
            region
            for region in body_regions
            if region.center[1] <= fallback.y + (fallback.h * 3.2)
        ]
        if not filtered_regions:
            return [fallback]
        return sorted(filtered_regions, key=lambda region: (region.y, region.x))

    def _choose_region_by_frame_elements(
        self,
        *,
        slide_number: int,
        focus: str,
        image_width: int,
        image_height: int,
        title_regions: list[TextRegion],
        body_regions: list[TextRegion],
        fallback: TextRegion,
    ) -> TextRegion | None:
        frame_elements = self.frame_elements_by_slide.get(slide_number, [])
        if not frame_elements:
            return None

        normalized_focus = focus.lower()
        title_region = self._choose_title_region(title_regions, fallback)
        body_sequence = self._body_region_sequence(body_regions, fallback)
        quoted_targets = self._extract_quoted_strings(focus)

        if slide_number == 1 and any(token in normalized_focus for token in ("title", "lecture notes", "artificial intelligence")):
            return TextRegion(
                x=int(image_width * 0.25),
                y=int(image_height * 0.16),
                w=int(image_width * 0.5),
                h=int(image_height * 0.12),
            )

        scored: list[tuple[float, int, FrameElement]] = []
        targets = quoted_targets or [focus]
        for element_index, element in enumerate(frame_elements):
            best_score = max(self._token_overlap_score(target, element.text) for target in targets)

            if element.kind == "frame_title" and any(token in normalized_focus for token in ("title", "header")):
                best_score += 1.2
            if element.kind == "block_title" and any(token in normalized_focus for token in ("header", "section", "definition", "interdisciplinary")):
                best_score += 0.8
            if element.kind == "bullet" and "bullet" in normalized_focus:
                best_score += 0.8
            if element.kind in ("block_body", "paragraph") and any(token in normalized_focus for token in ("definition", "paragraph", "text", "phrase")):
                best_score += 0.6

            if best_score > 0:
                scored.append((best_score, element_index, element))

        if not scored:
            return None

        _, element_index, best_element = max(scored, key=lambda item: (item[0], -item[1]))
        if best_element.kind == "frame_title":
            return title_region

        body_like = [element for element in frame_elements if element.kind != "frame_title"]
        body_index = 0
        for idx, element in enumerate(body_like):
            if element == best_element:
                body_index = idx
                break

        mapped_index = round((body_index / max(len(body_like) - 1, 1)) * (len(body_sequence) - 1))
        mapped_index = min(max(mapped_index, 0), len(body_sequence) - 1)
        return body_sequence[mapped_index]

    def _pick_region_for_focus(
        self,
        *,
        focus: str,
        item_index: int,
        total_items: int,
        title_regions: list[TextRegion],
        body_regions: list[TextRegion],
        fallback: TextRegion,
    ) -> TextRegion:
        normalized_focus = focus.lower()
        title_region = self._choose_title_region(title_regions, fallback)
        body_sequence = self._body_region_sequence(body_regions, fallback)

        if any(token in normalized_focus for token in ("title", "header")):
            return title_region

        if any(token in normalized_focus for token in ("diagram", "figure", "image", "chart", "table")):
            # For non-text references, pick the nearest large non-title region.
            return max(body_sequence, key=lambda region: region.area)

        if any(token in normalized_focus for token in ("definition", "paragraph", "text", "phrase", "bullet")):
            index = min(item_index, len(body_sequence) - 1)
            return body_sequence[index]

        if total_items == 1:
            return body_sequence[0]

        proportional_index = round((item_index / max(total_items - 1, 1)) * (len(body_sequence) - 1))
        proportional_index = min(max(proportional_index, 0), len(body_sequence) - 1)
        return body_sequence[proportional_index]

    def _ground_slide_focuses(
        self,
        *,
        slide_number: int,
        image_path: Path,
        items: list[dict[str, Any]],
    ) -> list[tuple[float, float]]:
        image = self._load_image(image_path)
        height, width = image.shape[:2]
        regions = self._detect_text_regions(image)

        fallback = TextRegion(
            x=int(width * 0.4),
            y=int(height * 0.35),
            w=max(1, int(width * 0.2)),
            h=max(1, int(height * 0.1)),
        )
        title_regions, body_regions = self._split_title_and_body_regions(regions, height)

        points: list[tuple[float, float]] = []
        for item_index, item in enumerate(items):
            focus = str(item.get("focus", "")).strip()
            normalized_focus = focus.lower()
            if slide_number == 1 and any(token in normalized_focus for token in ("title", "lecture notes", "artificial intelligence")):
                points.append((width * 0.5, height * 0.22))
                continue
            region = self._choose_region_by_frame_elements(
                slide_number=slide_number,
                focus=focus,
                image_width=width,
                image_height=height,
                title_regions=title_regions,
                body_regions=body_regions,
                fallback=fallback,
            )
            if region is None:
                region = self._pick_region_for_focus(
                    focus=focus,
                    item_index=item_index,
                    total_items=len(items),
                    title_regions=title_regions,
                    body_regions=body_regions,
                    fallback=fallback,
                )
            points.append(region.center)

        print(f"Grounded slide {slide_number} with {len(points)} cursor points.")
        return points

    def generate_cursor_incremental(
        self,
        subtitles_json: dict[str, Any],
        output_json_path: str,
    ) -> list[dict[str, Any]]:
        output_path = Path(output_json_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        slide_images = self._sorted_media_files(self.images_dir, (".png", ".jpg", ".jpeg", ".webp"))
        slide_audios = self._sorted_media_files(self.audio_dir, (".wav",))
        slide_keys = sorted(subtitles_json.keys(), key=self._extract_index)

        timeline: list[dict[str, Any]] = []
        global_time = 0.0

        for index, slide_key in enumerate(slide_keys):
            if index >= len(slide_images) or index >= len(slide_audios):
                raise ValueError(
                    f"Missing aligned slide image or audio for subtitle entry: {slide_key}"
                )

            slide = subtitles_json[slide_key]
            slide_number = int(slide.get("slide_number", index + 1))
            items = slide.get("items", [])
            if not items:
                global_time += self._get_audio_duration(slide_audios[index])
                continue

            image_path = slide_images[index]
            audio_path = slide_audios[index]
            audio_duration = self._get_audio_duration(audio_path)
            duration_per_item = audio_duration / max(len(items), 1)
            grounded_points = self._ground_slide_focuses(
                slide_number=slide_number,
                image_path=image_path,
                items=items,
            )

            for item_index, (item, point) in enumerate(zip(items, grounded_points), start=0):
                start_time = global_time + (item_index * duration_per_item)
                end_time = start_time + duration_per_item
                timeline.append(
                    {
                        "start": round(start_time, 3),
                        "end": round(end_time, 3),
                        "cursor": [round(point[0], 3), round(point[1], 3)],
                        "focus": str(item.get("focus", "")).strip(),
                    }
                )

            global_time += audio_duration
            with output_path.open("w", encoding="utf-8") as handle:
                json.dump(timeline, handle, indent=2, ensure_ascii=False)
            print(f"Saved cursor progress through slide {slide_number} -> {output_path}")

        return timeline

    def run(self, input_json_path: str, output_json_path: str) -> None:
        input_path = Path(input_json_path)
        if not input_path.exists():
            raise FileNotFoundError(f"Subtitle JSON not found: {input_path}")

        with input_path.open("r", encoding="utf-8") as handle:
            subtitle_data = json.load(handle)

        timeline = self.generate_cursor_incremental(subtitle_data, output_json_path)
        print(f"Cursor JSON saved at: {output_json_path} with {len(timeline)} entries.")


if __name__ == "__main__":
    agent = CursorAgent(
        images_dir="Data/intermediate/slide_images",
        audio_dir="Data/intermediate/slide_audios",
    )
    agent.run(
        input_json_path="Data/intermediate/subtitles.json",
        output_json_path="Data/intermediate/cursor.json",
    )
