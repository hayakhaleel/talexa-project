import os
import json
import time
import re
import base64
from pathlib import Path
from google import genai
from google.genai import types
import pdfplumber

API_KEY = os.environ.get("GEMINI_API_KEY")
if not API_KEY:
    raise EnvironmentError(
        "GEMINI_API_KEY environment variable is not set. "
        "Run: export GEMINI_API_KEY='your_key_here'"
    )

client = genai.Client(api_key=API_KEY)

MODELS_TO_TRY = [
    "gemini-2.5-flash",
    "gemini-2.5-flash-lite",
    "gemini-2.5-pro",
]

RETRY_429_DELAY = 15
MAX_429_RETRIES = 3

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp"}
MIME_MAP = {
    ".png":  "image/png",
    ".jpg":  "image/jpeg",
    ".jpeg": "image/jpeg",
    ".webp": "image/webp",
}


def get_pdf_from_input_folder(input_dir="Data/input"):
    folder = Path(input_dir)

    if not folder.exists():
        raise FileNotFoundError(f"Input folder not found: {input_dir}")

    pdf_files = list(folder.glob("*.pdf"))

    if not pdf_files:
        raise FileNotFoundError(f"No PDF files found in: {input_dir}")

    if len(pdf_files) > 1:
        print(f"Multiple PDFs found. Using first one: {pdf_files[0].name}")

    return str(pdf_files[0])


class MetaSimilarityEvaluator:
    def __init__(self, model_name: str = "gemini-2.5-flash"):
        self.model_id = model_name

    def extract_text_from_pdf(self, pdf_path: str) -> str:
        print(f"Extracting reference text from: {pdf_path}")
        full_text = ""
        try:
            with pdfplumber.open(pdf_path) as pdf:
                for page in pdf.pages:
                    text = page.extract_text()
                    if text:
                        full_text += text + "\n"
        except Exception as e:
            print(f"PDF Extraction Error: {e}")
        return full_text.strip()

    def load_slide_images(self, slides_dir: str) -> list:
        folder = Path(slides_dir)
        if not folder.is_dir():
            raise FileNotFoundError(f"Slide images folder not found: {slides_dir}")

        image_files = sorted(
            f for f in folder.iterdir()
            if f.suffix.lower() in IMAGE_EXTENSIONS
        )

        if not image_files:
            raise ValueError(
                f"No images ({', '.join(IMAGE_EXTENSIONS)}) found in: {slides_dir}"
            )

        print(f"Loading {len(image_files)} slide images from: {slides_dir}")

        parts = []
        for img_path in image_files:
            mime = MIME_MAP[img_path.suffix.lower()]
            data = base64.standard_b64encode(img_path.read_bytes()).decode("utf-8")
            parts.append(
                types.Part(
                    inline_data=types.Blob(mime_type=mime, data=data)
                )
            )

        print(f"  Loaded {len(parts)} slides.")
        return parts

    @staticmethod
    def _parse_json_response(raw: str) -> dict:
        cleaned = re.sub(r"^```(?:json)?\s*", "", raw.strip(), flags=re.IGNORECASE)
        cleaned = re.sub(r"\s*```$", "", cleaned)
        return json.loads(cleaned)

    def _query_model(self, model_name: str, contents, generation_config):
        for attempt in range(1, MAX_429_RETRIES + 1):
            try:
                response = client.models.generate_content(
                    model=model_name,
                    contents=contents,
                    config=generation_config,
                )
                if response.text:
                    return self._parse_json_response(response.text)
                return None

            except Exception as e:
                err_str = str(e)
                if "429" in err_str or "RESOURCE_EXHAUSTED" in err_str:
                    if attempt < MAX_429_RETRIES:
                        print(
                            f"  Quota hit on {model_name} "
                            f"(attempt {attempt}/{MAX_429_RETRIES}). "
                            f"Waiting {RETRY_429_DELAY}s..."
                        )
                        time.sleep(RETRY_429_DELAY)
                        continue
                    else:
                        print(f"  {model_name} quota exhausted after {attempt} attempts.")
                        raise
                raise

    def evaluate(
        self,
        slides_dir: str,
        source_text: str,
        output_path: str = "Evaluation/Results/meta_similarity_result.json",
    ):
        if not source_text:
            print("Error: No source text provided for comparison.")
            return None

        print(f"Starting MetaSimilarity evaluation for slides in: {slides_dir}")

        slide_parts = self.load_slide_images(slides_dir)

        user_query = (
            f"REFERENCE MATERIAL (Original Source Content):\n{source_text}\n\n"
            "Your task is to perform 'MetaSimilarity' analysis. You are provided with "
            "a sequence of educational slide images. Compare them against the original "
            "source reference material.\n"
            "Focus on high-level semantic meaning and educational precision, not just "
            "word-for-word matching.\n\n"
            "Please evaluate the following:\n"
            "1. Semantic Alignment: Do the slides capture the core concepts of the "
            "reference material? (Score 0.0 - 1.0)\n"
            "2. Precision: Are technical terms and relationships between concepts "
            "handled accurately? (Score 0.0 - 1.0)\n"
            "3. Hallucination Check: Did the slides introduce any information NOT "
            "present in the reference material?\n"
            "4. Missing Information: What key concepts from the reference were lost "
            "in the slides?\n\n"
            "Respond ONLY with a JSON object — no markdown fences, no extra text — "
            "with keys: 'semantic_score', 'precision_score', 'hallucinations', "
            "'omissions', and 'analysis'."
        )

        generation_config = types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema={
                "type": "object",
                "properties": {
                    "semantic_score":  {"type": "number", "description": "0.0-1.0"},
                    "precision_score": {"type": "number", "description": "0.0-1.0"},
                    "hallucinations": {
                        "type": "array",
                        "items": {"type": "string"},
                    },
                    "omissions": {
                        "type": "array",
                        "items": {"type": "string"},
                    },
                    "analysis": {"type": "string"},
                },
                "required": [
                    "semantic_score",
                    "precision_score",
                    "hallucinations",
                    "omissions",
                    "analysis",
                ],
            },
        )

        contents = [
            types.Content(
                role="user",
                parts=slide_parts + [types.Part(text=user_query)],
            )
        ]

        models_to_try = [self.model_id] + [
            m for m in MODELS_TO_TRY if m != self.model_id
        ]

        result = None
        for model_name in models_to_try:
            print(f"Querying {model_name}...")
            try:
                result = self._query_model(model_name, contents, generation_config)
                if result:
                    print(f"Got a valid response from {model_name}.")
                    break
            except json.JSONDecodeError as e:
                print(f"{model_name} returned invalid JSON: {e}")
            except Exception as e:
                print(f"{model_name} failed: {e}")

        if not result:
            print(
                "All Gemini model attempts failed.\n"
                "Your free daily quota may be exhausted — try again tomorrow, or\n"
                "add billing at https://aistudio.google.com/ for higher limits."
            )
            return None

        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=4, ensure_ascii=False)

        print(f"MetaSimilarity evaluation complete. Results saved to '{output_path}'")
        return result


if __name__ == "__main__":
    SLIDES_DIR = "Data/intermediate/slide_images"

    try:
        REFERENCE_PDF = get_pdf_from_input_folder()
    except Exception as e:
        print(e)
        exit(1)

    evaluator = MetaSimilarityEvaluator(model_name="gemini-2.5-flash")

    missing = False
    if not os.path.exists(SLIDES_DIR):
        print(f"Missing slide images folder: {SLIDES_DIR}")
        missing = True

    if not missing:
        source_content = evaluator.extract_text_from_pdf(REFERENCE_PDF)
        evaluator.evaluate(SLIDES_DIR, source_content)
