import os
import json
import time
import re
import base64
import requests
from pathlib import Path
from google import genai
from google.genai import types


API_KEY = os.environ.get("GEMINI_API_KEY")
if not API_KEY:
    raise EnvironmentError("Missing GEMINI_API_KEY")

client = genai.Client(api_key=API_KEY)

GEMINI_PRIMARY   = "gemini-2.5-flash"
GEMINI_FALLBACKS = ["gemini-2.5-flash-lite", "gemini-2.5-pro"]

OLLAMA_BASE_URL  = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
QWEN_TEXT        = "qwen2.5:7b"
OLLAMA_TIMEOUT   = int(os.environ.get("OLLAMA_TIMEOUT", "120"))

RETRY_429_DELAY  = 15
MAX_429_RETRIES  = 3

VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".webm"}
VIDEO_MIME_MAP   = {
    ".mp4":  "video/mp4",
    ".mov":  "video/quicktime",
    ".avi":  "video/x-msvideo",
    ".mkv":  "video/x-matroska",
    ".webm": "video/webm",
}


def get_pdf_from_output_folder(output_dir="Data/output"):
    folder = Path(output_dir)
    if not folder.exists():
        raise FileNotFoundError(f"Output folder not found: {output_dir}")
    pdfs = [f for f in folder.iterdir() if f.suffix.lower() == ".pdf"]
    if not pdfs:
        raise FileNotFoundError(f"No PDF found in: {output_dir}")
    if len(pdfs) > 1:
        print(f"Multiple PDFs found — using: {pdfs[0].name}")
    return str(pdfs[0])


def get_video_from_output_folder(output_dir="Data/output"):
    folder = Path(output_dir)
    if not folder.exists():
        raise FileNotFoundError(f"Output folder not found: {output_dir}")
    videos = [f for f in folder.iterdir() if f.suffix.lower() in VIDEO_EXTENSIONS]
    if not videos:
        raise FileNotFoundError(f"No video files found in: {output_dir}")
    if len(videos) > 1:
        print(f"Multiple videos found — using: {videos[0].name}")
    return str(videos[0])


def _parse_json(text: str):
    text = re.sub(r"^```(json)?", "", text.strip())
    text = re.sub(r"```$",        "", text.strip())
    return json.loads(text.strip())


def extract_text_from_pdf(pdf_path: str, gemini_client, gemini_model: str) -> str:
    """
    Render every PDF page as an image with PyMuPDF, then send all pages
    to Gemini vision in one call to read the actual slide content.
    This handles image-based / scanned PDFs where there is no embedded text.

    Install: pip install pymupdf
    """
    try:
        import fitz  # PyMuPDF
    except ImportError:
        raise ImportError(
            "PyMuPDF is required for PDF extraction.\n"
            "Install it with:  pip install pymupdf"
        )

    path = Path(pdf_path)
    if not path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    doc = fitz.open(str(path))
    image_parts = []

    for i, page in enumerate(doc, start=1):
        # Render at 150 DPI — enough for Gemini to read text clearly
        pix  = page.get_pixmap(dpi=150)
        data = base64.b64encode(pix.tobytes("png")).decode("utf-8")
        image_parts.append(
            types.Part(inline_data=types.Blob(mime_type="image/png", data=data))
        )

    doc.close()
    print(f"  Rendered {len(image_parts)} slide(s) from '{path.name}' — sending to Gemini OCR...")

    prompt = (
        "You are reading a set of presentation slides. "
        "For each slide image, transcribe ALL visible text exactly as it appears: "
        "titles, bullet points, labels, numbers, captions — everything. "
        "Separate each slide with a header line: '--- Slide N ---'. "
        "Do NOT summarise, paraphrase, or skip any slide."
    )

    contents = [
        types.Content(
            role="user",
            parts=image_parts + [types.Part(text=prompt)]
        )
    ]

    for attempt in range(MAX_429_RETRIES):
        try:
            res = gemini_client.models.generate_content(
                model=gemini_model,
                contents=contents,
                config=types.GenerateContentConfig(),
            )
            if res.text:
                print(f"  OCR complete — {len(res.text)} chars extracted.")
                return res.text
        except Exception as e:
            if "429" in str(e):
                print(f"  OCR rate-limited, waiting {RETRY_429_DELAY}s...")
                time.sleep(RETRY_429_DELAY)
                continue
            raise

    raise RuntimeError("Gemini OCR failed to extract text from the PDF.")


class PresentQuizEvaluator:

    def __init__(self, num_questions=4):
        self.num_questions = num_questions

    def _load_video_part(self, video_path: str):
        """Return a video as a Gemini Part for multimodal answering."""
        path = Path(video_path)
        mime = VIDEO_MIME_MAP[path.suffix.lower()]
        data = base64.b64encode(path.read_bytes()).decode("utf-8")
        return types.Part(inline_data=types.Blob(mime_type=mime, data=data))

    def _call_ollama(self, prompt: str, label: str):
        url     = f"{OLLAMA_BASE_URL}/api/chat"
        payload = {
            "model":    QWEN_TEXT,
            "messages": [{"role": "user", "content": prompt}],
            "stream":   False,
            "options":  {"temperature": 0.2},
        }
        print(f"  [{label}] calling {QWEN_TEXT} via Ollama...")
        try:
            resp = requests.post(url, json=payload, timeout=OLLAMA_TIMEOUT)
            if not resp.ok:
                print(f"  [{label}] HTTP {resp.status_code}: {resp.text[:400]}")
                resp.raise_for_status()
            return _parse_json(resp.json()["message"]["content"])
        except requests.exceptions.ConnectionError:
            print(f"  [{label}] Cannot reach Ollama at {OLLAMA_BASE_URL} — is it running?")
        except requests.exceptions.Timeout:
            print(f"  [{label}] Timed out after {OLLAMA_TIMEOUT}s.")
        except Exception as e:
            print(f"  [{label}] Error: {e}")
        return None

    def _call_gemini(self, contents, config, label: str):
        for model in [GEMINI_PRIMARY] + GEMINI_FALLBACKS:
            print(f"  [{label}] trying {model}...")
            for _ in range(MAX_429_RETRIES):
                try:
                    res = client.models.generate_content(
                        model=model, contents=contents, config=config
                    )
                    if res.text:
                        return _parse_json(res.text)
                except Exception as e:
                    if "429" in str(e):
                        print(f"  [{label}] rate-limited, waiting {RETRY_429_DELAY}s...")
                        time.sleep(RETRY_429_DELAY)
                        continue
                    print(f"  [{label}] {model} error: {e}")
                    break
        return None

    def generate_questions(self, slide_text: str) -> list:
        """
        qwen2.5:7b reads the extracted slide text and generates
        questions that a presenter should cover when explaining these slides.
        """
        print(f"\n[Phase 1] Generating {self.num_questions} questions with {QWEN_TEXT}...")

        prompt = f"""You are an educational quiz creator reviewing presentation slide content.
Based on the slides below, generate exactly {self.num_questions} factual questions \
that a presenter SHOULD be able to answer if they properly explained these slides.

SLIDE CONTENT:
{slide_text}

Rules:
- Every question must be answerable from the slide content above.
- Prefer specific facts: definitions, numbers, steps, names, comparisons.
- Each answer should be concise (a word, number, or short phrase).
- Return ONLY a valid JSON array — no explanation, no markdown fences.

Format:
[
  {{"question": "...", "answer": "..."}}
]"""

        result = self._call_ollama(prompt, "Q-Gen")
        if not result:
            raise RuntimeError(
                f"Question generation failed. "
                f"Is '{QWEN_TEXT}' pulled? Run: ollama pull {QWEN_TEXT}"
            )
        print(f"  Generated {len(result)} question(s).")
        return result

    def answer_questions(self, questions: list, video_path: str) -> list:
        """
        Gemini watches the video and answers each question based solely
        on what the presenter said — testing if they explained their slides.
        """
        print(f"\n[Phase 2] Answering questions from video with Gemini ({GEMINI_PRIMARY})...")

        video_part = self._load_video_part(video_path)
        q_block    = "\n".join(f"{i+1}. {q['question']}" for i, q in enumerate(questions))

        prompt = f"""Watch the video carefully and answer each question based ONLY on \
what the presenter says or shows in the video.
Be concise — a word, number, or short phrase per answer.
If the presenter did not cover a topic, write "not mentioned".

QUESTIONS:
{q_block}

Return ONLY a valid JSON array — no explanation, no markdown fences.

Format:
[
  {{"question_number": 1, "answer": "..."}}
]"""

        config   = types.GenerateContentConfig(response_mime_type="application/json")
        contents = [
            types.Content(role="user", parts=[video_part, types.Part(text=prompt)])
        ]

        answers = self._call_gemini(contents, config, "Q-A")
        if not answers:
            raise RuntimeError("Answering phase failed across all Gemini models.")

        ans_map  = {a["question_number"]: a["answer"] for a in answers}
        enriched = [
            {**q, "vlm_answer": ans_map.get(i + 1, "not mentioned")}
            for i, q in enumerate(questions)
        ]
        return enriched

    def grade(self, qa_pairs: list) -> list:
        """
        qwen2.5:7b compares what the slides expected (correct answer)
        vs what the presenter said in the video (vlm_answer).
        """
        print(f"\n[Phase 3] Grading with {QWEN_TEXT}...")

        items = json.dumps(
            [
                {
                    "question":  q["question"],
                    "correct":   q["answer"],
                    "predicted": q["vlm_answer"],
                }
                for q in qa_pairs
            ],
            indent=2,
        )

        prompt = f"""You are a strict but fair grader evaluating whether a presenter \
properly explained their slides.
For each item, score how well the predicted answer matches the correct answer:
  1.0 — fully correct or semantically equivalent
  0.5 — partially correct or mentioned vaguely
  0.0 — wrong, missing, or "not mentioned"

{items}

Return ONLY a valid JSON array — no explanation, no markdown fences.

Format:
[
  {{"question_number": 1, "score": 0.0}}
]"""

        grades = self._call_ollama(prompt, "Grade")
        if not grades:
            raise RuntimeError(
                f"Grading failed. Is '{QWEN_TEXT}' pulled? Run: ollama pull {QWEN_TEXT}"
            )

        gmap = {g["question_number"]: g["score"] for g in grades}
        for i, q in enumerate(qa_pairs):
            q["score"] = gmap.get(i + 1, 0.0)
        return qa_pairs

    def evaluate(self, pdf_path: str, video_path: str):
        """
        pdf_path   — path to the slides PDF (text extracted locally, no vision needed)
        video_path — path to the presentation video (Gemini watches this)
        """
        print("\n[Step 0] Extracting slide text from PDF...")
        slide_text = extract_text_from_pdf(pdf_path, client, GEMINI_PRIMARY)

        questions = self.generate_questions(slide_text)

        qa = self.answer_questions(questions, video_path)

        graded = self.grade(qa)

        total_score = sum(q["score"] for q in graded)
        max_score   = self.num_questions

        os.makedirs("Evaluation/Results", exist_ok=True)
        output_path = "Evaluation/Results/present_quiz_result.json"

        result = {
            "total_score": total_score,
            "max_score":   max_score,
            "questions":   graded,
        }

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)

        print(f"\nSaved  → {output_path}")
        print(f"Score  → {total_score}/{max_score}")
        return result

if __name__ == "__main__":
    evaluator = PresentQuizEvaluator(num_questions=4)

    try:
        video_path = get_video_from_output_folder()
    except Exception as e:
        print(f"Video error: {e}")
        video_path = None

    try:
        pdf_path = get_pdf_from_output_folder()
    except Exception as e:
        print(f"PDF error: {e}")
        pdf_path = None

    if video_path and pdf_path:
        evaluator.evaluate(pdf_path=pdf_path, video_path=video_path)
    else:
        print("Cannot evaluate — need both a PDF and a video in Data/output/.")
