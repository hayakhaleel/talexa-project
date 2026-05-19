import os
import sys
import json
import re #regex library
import copy

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class JsonTranslateAgent:

    def __init__(
        self,
        model_name="facebook/nllb-200-distilled-600M",
        base_data_dir="Data",
        source_lang="eng_Latn",
        target_lang="arb_Arab",
        max_length=256,
    ):
        self.model_name = model_name
        self.base_data_dir = base_data_dir
        self.source_lang = source_lang
        self.target_lang = target_lang
        self.max_length = max_length
        self.tokenizer = None
        self.model = None

    #loads the subtitle,json file
    def load_json(self, json_path):
        if not os.path.exists(json_path):
            raise FileNotFoundError(f"JSON file not found: {json_path}")

        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        print(f"Loaded JSON from: {json_path}")
        return data

    #counts how many "sentence" feilds exists / used for validation
    def count_sentence_fields(self, data):
        if isinstance(data, dict):
            count = 1 if "sentence" in data else 0
            return count + sum(self.count_sentence_fields(value) for value in data.values())

        if isinstance(data, list):
            return sum(self.count_sentence_fields(item) for item in data)

        return 0

    #validation to make sure that the translation is arabic
    def _contains_arabic(self, text):
        return bool(re.search(r"[\u0600-\u06FF]", str(text)))

    #checks if the text contains chineese korean or japaneese characters
    def _contains_cjk(self, text):
        return bool(re.search(r"[\u3040-\u30FF\u3400-\u4DBF\u4E00-\u9FFF\uF900-\uFAFF]", str(text)))

    #counts the english letters , to make sure we can translate it
    def _latin_letter_count(self, text):
        return len(re.findall(r"[A-Za-z]", str(text)))

    #validation to make sure the output is arabic
    def validate_arabic_translation(self, source, translated, location="root"):
        errors = [] #creates a list called erroe

        if isinstance(source, dict): #checks if the original subtitles is a dict
            if not isinstance(translated, dict):
                return [f"{location}: expected object, got {type(translated).__name__}"]

            #checks if the "focus prompt" has been altered
            for metadata_key in ("focus", "image"):
                if metadata_key in source and translated.get(metadata_key) != source.get(metadata_key):
                    errors.append(f"{location}.{metadata_key}: metadata was modified")

            #checks if sentence feild exists
            if "sentence" in source:
                sentence_location = f"{location}.sentence"
                translated_sentence = translated.get("sentence")

                if not isinstance(translated_sentence, str):
                    errors.append(f"{sentence_location}: translated value is not a string")
                #now validates the output if is valid
                else:
                    if self._contains_cjk(translated_sentence):
                        errors.append(f"{sentence_location}: contains Chinese/Japanese/Korean characters")
                    if self._latin_letter_count(source.get("sentence", "")) >= 3 and not self._contains_arabic(translated_sentence):
                        errors.append(f"{sentence_location}: does not contain Arabic text")

            #checks if all keys exist
            for key, value in source.items():
                if key == "sentence":
                    continue
                if key not in translated:
                    errors.append(f"{location}.{key}: missing key")
                    continue
                errors.extend(self.validate_arabic_translation(value, translated[key], f"{location}.{key}"))

            return errors

        if isinstance(source, list):
            if not isinstance(translated, list):
                return [f"{location}: expected list, got {type(translated).__name__}"]
            if len(source) != len(translated): #checks if the number of focus/sentence match
                errors.append(f"{location}: list length changed from {len(source)} to {len(translated)}")

            for idx, (source_item, translated_item) in enumerate(zip(source, translated)):
                errors.extend(self.validate_arabic_translation(source_item, translated_item, f"{location}[{idx}]"))

        return errors

    def chunk_top_level_json(self, json_data, chunk_size=3):
        if not isinstance(json_data, dict): #chceck if dictionary
            return [json_data]

        items = list(json_data.items()) #convert the dictionary to list
        chunks = []

        for i in range(0, len(items), chunk_size): #takes every 3 items and puts them in one chunk
            chunk = dict(items[i:i + chunk_size])
            chunks.append(chunk)

        return chunks

    def _load_translation_model(self): #load translation model
        if self.tokenizer is not None and self.model is not None:
            return

        try:
            import torch
            from transformers import AutoModelForSeq2SeqLM, AutoTokenizer #loads model if exists
        except ImportError as exc: #if error, then send error message
            raise ImportError(
                "NLLB translation requires the 'transformers' and 'torch' packages. "
                "Install them before running Arabic translation."
            ) from exc

        #load the model
        print(f"Loading translation model: {self.model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)
        self.model.eval()

    def translate_sentence(self, sentence):
        text = str(sentence).strip()  #makes sure its string and remove white spaces
        if not text:
            return ""

        self._load_translation_model() #calls the model
        self.tokenizer.src_lang = self.source_lang #sets the source language to english

        inputs = self.tokenizer(
            text,
            return_tensors="pt", #pytorch tesors
            truncation=True,
            max_length=self.max_length, #256
        ).to(self.device) #move the tokens to the GPU

        forced_bos_token_id = self.tokenizer.convert_tokens_to_ids(self.target_lang) #forces arabic translation
        output = self.model.generate(
            **inputs,
            forced_bos_token_id=forced_bos_token_id,
            max_length=self.max_length,
        )
    #returns the decoded text translated
        return self.tokenizer.decode(output[0], skip_special_tokens=True).strip()

    
    def translate_sentence_fields(self, data):
        if isinstance(data, dict):
            translated = {}
            for key, value in data.items(): #actual translation 
                if key == "sentence":
                    translated[key] = self.translate_sentence(value)
                else:
                    translated[key] = self.translate_sentence_fields(value)
            return translated

        if isinstance(data, list):
            return [self.translate_sentence_fields(item) for item in data]
        #if it is not a dict or lst, copy it wirhout translating
        return copy.deepcopy(data)

    #controls the entire translation
    def translate_json(self, json_data, max_attempts=3):
        expected_sentence_count = self.count_sentence_fields(json_data) #counts sentences

        #calls translate sentence feilds, then counts the translated output
        for attempt in range(1, max_attempts + 1):
            print(f"Translating sentence fields with NLLB... attempt {attempt}/{max_attempts}")
            translated_json = self.translate_sentence_fields(json_data)
            translated_sentence_count = self.count_sentence_fields(translated_json)

            #f they are not the same count , then not valid repeat
            if translated_sentence_count != expected_sentence_count:
                if attempt == max_attempts:
                    raise ValueError(
                        "Translation output did not preserve the number of sentence fields. "
                        f"Expected {expected_sentence_count}, got {translated_sentence_count}."
                    )
                continue

            #if they are valid, validate the content of the translation
            validation_errors = self.validate_arabic_translation(json_data, translated_json)
            if validation_errors:
                print("Translation validation failed:")
                for error in validation_errors[:10]:
                    print(f" - {error}")
                if attempt == max_attempts:
                    raise ValueError(
                        "Translation output failed Arabic validation:\n"
                        + "\n".join(validation_errors[:20])
                    )
                continue

            print("Translation completed successfully.")
            return translated_json

        raise ValueError("Translation failed after all retry attempts.")

    #functon for chunking
    def translate_json_in_chunks(self, json_data, chunk_size=3, max_attempts=3):
        chunks = self.chunk_top_level_json(json_data, chunk_size=chunk_size)
        #if there is only one chunk, just translate it directly
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
    agent = JsonTranslateAgent(model_name="facebook/nllb-200-distilled-600M")

    result_path = agent.run(
        input_json_path=r"Data/input/lecture1_sentences.json",
        output_json_path=r"C:\Users\user\Desktop\Talexa\Data\Intermediate\lecture1_sentences_arabic.json"
    )

    print(f"\nFinal translated JSON file: {result_path}")
