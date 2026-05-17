import os
import re
import sys
import json
from pathlib import Path
from typing import List, Dict, Any, Optional

import ollama

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from Prompts.subtitle_focus_prompt import SUBTITLE_FOCUS_PROMPT


class SubtitleFocusAgent:

    def __init__(
        self,
        model_name="qwen3-vl:8b",
        output_path="Data/intermediate/subtitles.json",
        max_retries=3,
    ):
        self.model_name = model_name
        self.output_path = output_path
        self.max_retries = max_retries
        self.system_prompt = SUBTITLE_FOCUS_PROMPT.strip()
        self.client = ollama.Client()

        self.banned_patterns = [
            r"^this slide",
            r"^the slide",
            r"^here we see",
            r"^it mentions",
            r"^the date",
            r"^the location",
            r"^the definition is provided",
            r"^the slide is titled",
        ]

        self.default_fallback = [{
            "sentence": "This slide introduces an important concept which is explained through the visible content.",
            "focus": "main title and central slide content"
        }]

    def _extract_slide_number(self, file_path):
        name = os.path.basename(file_path)
        nums = re.findall(r"\d+", name)
        return int(nums[-1]) if nums else 0

    def get_slide_images(self, folder):
        if not os.path.exists(folder):
            raise FileNotFoundError(folder)

        images = []

        for f in os.listdir(folder):
            if f.lower().endswith((".png", ".jpg", ".jpeg", ".webp")):
                images.append(os.path.join(folder, f))

        images.sort(key=self._extract_slide_number)
        return images

    def _is_bad_sentence(self, sentence):
        s = sentence.lower().strip()

        if len(sentence.split()) < 6:
            return True

        for pattern in self.banned_patterns:
            if re.match(pattern, s):
                return True

        return False

    def _clean_json(self, text):
        text = text.strip()
        text = re.sub(r"^```(?:json)?", "", text)
        text = re.sub(r"```$", "", text)

        start = text.find("{")
        end = text.rfind("}")

        if start != -1 and end != -1:
            text = text[start:end + 1]

        return text

    def _looks_like_fallback_or_generic(self, items: List[Dict[str, str]]) -> bool:
        if not items:
            return True

        if len(items) == 1:
            s = items[0].get("sentence", "").strip().lower()
            f = items[0].get("focus", "").strip().lower()

            if s == self.default_fallback[0]["sentence"].lower():
                return True

            generic_starts = [
                "this slide introduces",
                "this slide explains",
                "this slide presents",
                "an important concept is explained",
            ]

            if any(s.startswith(prefix) for prefix in generic_starts):
                return True

            if f in {
                "main title and central slide content",
                "main title",
                "central slide content",
                "title and main content",
            }:
                return True

        return False

    def parse_output(self, raw_text) -> Optional[List[Dict[str, str]]]:
        cleaned = self._clean_json(raw_text)

        try:
            data = json.loads(cleaned)
            valid_items = []

            for item in data.get("sentences", []):
                sentence = item.get("sentence", "").strip()
                focus = item.get("focus", "").strip()

                if not sentence or not focus:
                    continue

                if self._is_bad_sentence(sentence):
                    continue

                valid_items.append({
                    "sentence": sentence,
                    "focus": focus
                })

            if not valid_items:
                return None

            return valid_items

        except Exception:
            return None

    def generate_for_slide(self, image_path, slide_index):
        print(f"Processing slide {slide_index}")

        last_raw = None

        for attempt in range(1, self.max_retries + 1):
            print(f"  Attempt {attempt}/{self.max_retries}")

            extra_instruction = ""
            if attempt > 1:
                extra_instruction = (
                    " Your previous answer was too generic or invalid. "
                    "Return only valid JSON with a 'sentences' list. "
                    "Each sentence must sound like a lecturer explanation, "
                    "not a slide description, and each focus must point to a specific visible region or phrase."
                )

            response = self.client.chat(
                model=self.model_name,
                messages=[
                    {
                        "role": "system",
                        "content": self.system_prompt
                    },
                    {
                        "role": "user",
                        "content": f"Analyze slide {slide_index} and generate lecture explanation.{extra_instruction}",
                        "images": [image_path]
                    }
                ]
            )

            raw = response["message"]["content"]
            last_raw = raw

            parsed = self.parse_output(raw)

            if parsed is None:
                print("  Parsed output invalid, retrying...")
                continue

            if self._looks_like_fallback_or_generic(parsed):
                print("  Output too generic, retrying...")
                continue

            print("  Accepted.")
            return parsed

        print(f"  All attempts failed for slide {slide_index}. Using fallback.")
        if last_raw:
            print("  Last raw response preview:")
            print(last_raw[:300])

        return self.default_fallback

    def run(self, slide_folder):
        images = self.get_slide_images(slide_folder)
        results = {}

        for i, img in enumerate(images, start=1):
            items = self.generate_for_slide(img, i)

            results[f"slide_{i}"] = {
                "slide_number": i,
                "image": img,
                "items": items
            }

        Path(os.path.dirname(self.output_path)).mkdir(parents=True, exist_ok=True)

        with open(self.output_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

        print("\nSaved:", self.output_path)


if __name__ == "__main__":
    agent = SubtitleFocusAgent()
    agent.run("Data/intermediate/lecture1_slides")
