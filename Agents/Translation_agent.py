import os
import sys
import json
import ollama

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from Prompts.Translation_prompt import JSON_TRANSLATE_PROMPT


class JsonTranslateAgent:

    def __init__(self, model_name="qwen2.5:7b", base_data_dir="Data"):
        self.model_name = model_name
        self.base_data_dir = base_data_dir

    def load_json(self, json_path):
        if not os.path.exists(json_path):
            raise FileNotFoundError(f"JSON file not found: {json_path}")

        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        print(f"Loaded JSON from: {json_path}")
        return data

    def count_sentence_fields(self, data):
        if isinstance(data, dict):
            count = 1 if "sentence" in data else 0
            return count + sum(self.count_sentence_fields(value) for value in data.values())

        if isinstance(data, list):
            return sum(self.count_sentence_fields(item) for item in data)

        return 0

    def chunk_top_level_json(self, json_data, chunk_size=3):
        if not isinstance(json_data, dict):
            return [json_data]

        items = list(json_data.items())
        chunks = []

        for i in range(0, len(items), chunk_size):
            chunk = dict(items[i:i + chunk_size])
            chunks.append(chunk)

        return chunks

    def translate_json(self, json_data, max_attempts=3):
        expected_sentence_count = self.count_sentence_fields(json_data)

        for attempt in range(1, max_attempts + 1):
            print(f"Sending JSON to translation model... attempt {attempt}/{max_attempts}")

            response = ollama.chat(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": JSON_TRANSLATE_PROMPT},
                    {
                        "role": "user",
                        "content": json.dumps(json_data, ensure_ascii=False, indent=2)
                    },
                ],
            )

            translated_text = response["message"]["content"].strip()

            try:
                translated_json = json.loads(translated_text)
            except json.JSONDecodeError:
                print("Model did not return valid JSON.")
                if attempt == max_attempts:
                    raise ValueError(f"Invalid JSON returned by model:\n{translated_text}")
                continue

            translated_sentence_count = self.count_sentence_fields(translated_json)

            if translated_sentence_count != expected_sentence_count:
                print(
                    "Sentence field count mismatch. "
                    f"Expected {expected_sentence_count}, got {translated_sentence_count}."
                )
                if attempt == max_attempts:
                    raise ValueError(
                        "Translation output did not preserve the number of sentence fields. "
                        f"Expected {expected_sentence_count}, got {translated_sentence_count}."
                    )
                continue

            print("Translation completed successfully.")
            return translated_json

        raise ValueError("Translation failed after all retry attempts.")

    def translate_json_in_chunks(self, json_data, chunk_size=3, max_attempts=3):
        chunks = self.chunk_top_level_json(json_data, chunk_size=chunk_size)

        if len(chunks) == 1:
            return self.translate_json(json_data, max_attempts=max_attempts)

        combined_result = {}
        total_chunks = len(chunks)

        for idx, chunk in enumerate(chunks, start=1):
            expected_sentence_count = self.count_sentence_fields(chunk)
            print(
                f"Translating chunk {idx}/{total_chunks} "
                f"with {len(chunk)} top-level entries and {expected_sentence_count} sentence fields..."
            )

            translated_chunk = self.translate_json(chunk, max_attempts=max_attempts)
            combined_result.update(translated_chunk)

        return combined_result

    def save_json(self, json_data, output_json_path):
        os.makedirs(os.path.dirname(output_json_path) or ".", exist_ok=True)

        with open(output_json_path, "w", encoding="utf-8") as f:
            json.dump(json_data, f, ensure_ascii=False, indent=2)

        print(f"Translated JSON saved to: {output_json_path}")

    def run(self, input_json_path, output_json_path=None, chunk_size=3):
        if not os.path.exists(input_json_path):
            raise FileNotFoundError(f"JSON file not found: {input_json_path}")

        json_name = os.path.splitext(os.path.basename(input_json_path))[0]

        if output_json_path is None:
            out_dir = os.path.join(self.base_data_dir, "output")
            os.makedirs(out_dir, exist_ok=True)
            output_json_path = os.path.join(out_dir, f"{json_name}_arabic.json")
        else:
            os.makedirs(os.path.dirname(output_json_path) or ".", exist_ok=True)

        json_data = self.load_json(input_json_path)
        translated_json = self.translate_json_in_chunks(json_data, chunk_size=chunk_size)
        self.save_json(translated_json, output_json_path)

        print("JSON translate agent completed successfully.")
        return output_json_path


if __name__ == "__main__":
    agent = JsonTranslateAgent(model_name="qwen2.5:7b")

    result_path = agent.run(
        input_json_path=r"Data/input/lecture1_sentences.json",
        output_json_path=r"C:\Users\user\Desktop\Talexa\Data\Intermediate\lecture1_sentences_arabic.json"
    )

    print(f"\nFinal translated JSON file: {result_path}")