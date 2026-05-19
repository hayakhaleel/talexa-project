import os
import re
import json
import asyncio #runs the slide pipepline synchronously
import hashlib #hashes the audio
from pathlib import Path
from typing import List, Dict, Any, Optional

import requests
from elevenlabs import save
from elevenlabs.client import ElevenLabs
from pydub import AudioSegment #combines audio chucnks and adds whitenoise between audios


class SpeechAgent:

    def __init__(
        self,
        subtitles_json_path: str = "Data/intermediate/final_subtitles.json", #path of  input 
        output_dir: str = "Data/intermediate/speech_output", #path for output
        ref_audio_path: Optional[str] = "Data/input/ref_clean.wav", #path of ref audio
        api_key: Optional[str] = None,
        user_voice_name: str = "talexa_user_voice", #name of voice clones
        voice_cache_path: str = "Data/intermediate/voice_cache.json", #stores in cache the id
        fallback_voice_id: Optional[str] = None, #if audio failes, use backup
        model_id: str = "eleven_multilingual_v2", #TTS model used
        chunk_max_len: int = 250, #maximum num of characters in each audio
        silence_ms: int = 150, # 1.50 second silence between audios
    ):
        self.subtitles_json_path = subtitles_json_path
        self.output_dir = output_dir
        self.ref_audio_path = ref_audio_path
        self.api_key = api_key or "sk_0f4715ea095118af6803da9ad114513bc59d53392bea09d1"
        self.user_voice_name = user_voice_name.strip()
        self.voice_cache_path = voice_cache_path
        self.fallback_voice_id = fallback_voice_id
        self.model_id = model_id
        self.chunk_max_len = chunk_max_len
        self.silence_ms = silence_ms

        if not self.api_key:
            raise ValueError(
                "Missing ElevenLabs API key. Set ELEVENLABS_API_KEY in your environment."
            )

        self.client = ElevenLabs(api_key=self.api_key) #creates a client
        self.base_headers = {"xi-api-key": self.api_key} #creates headers to send the API request

        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(os.path.dirname(self.voice_cache_path), exist_ok=True)

        #prints statements
        print("[SpeechAgent Ready]")
        print(f"[Subtitles File] {self.subtitles_json_path}")
        print(f"[Reference Audio] {self.ref_audio_path}")
        print(f"[Output Dir] {self.output_dir}")

    def _load_voice_cache(self) -> Dict[str, Any]:
        if not os.path.exists(self.voice_cache_path): #checks if the voice cache file exists.
            return {}
        try:
            with open(self.voice_cache_path, "r", encoding="utf-8") as f:
                data = json.load(f) #reads the cache file
            return data if isinstance(data, dict) else {} #if exists load it to a dictionary
        except Exception:
            return {} #if it doesnt exist, return empty 

    """ EXAMPLE CACHE FILE
{
  "abc123hash": {
    "voice_id": "EXAMPLE_VOICE_ID",
    "voice_name": "talexa_user_voice",
    "audio_path": "Data/input/ref_clean.wav"
  }
}

    """

    #if voice cache doesnt exist, save one into a json file
    def _save_voice_cache(self, cache: Dict[str, Any]) -> None:
        with open(self.voice_cache_path, "w", encoding="utf-8") as f:
            json.dump(cache, f, indent=2, ensure_ascii=False) #creates a json file

        #hahes the file 
    def _hash_file(self, path: str) -> str:
        sha = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(1024 * 1024), b""):
                sha.update(chunk)
        return sha.hexdigest()

    #cleans the subtitles, so it will be able to transfom TTS
    def clean_text(self, text: str) -> str:
        if text is None:
            return ""

        text = str(text)
        text = text.replace("\n", " ")
        text = text.replace("•", " ")
        text = text.replace("—", " ")
        text = text.replace("–", " ")
        text = text.replace("→", " to ")
        text = text.replace("&", " and ")

        #removes any characters that are not arabic or english or any basic symbols
        text = re.sub(r"[^\u0600-\u06FFA-Za-z0-9\s\.\?!،,:;'\-]", "", text)
        text = re.sub(r"\s+", " ", text).strip()
        return text

    #splits the subtitle file into chunks
    def split_text(self, text: str, max_len: Optional[int] = None) -> List[str]:
        max_len = max_len or self.chunk_max_len
        text = self.clean_text(text)

        if not text:
            return []
        #if the entire text is less than 250 characters, just send the text
        if len(text) <= max_len:
            return [text]

        #split the text after any ending punctuhations
        sentences = re.split(r"(?<=[\.\!\?؟])\s+", text)
        sentences = [s.strip() for s in sentences if s.strip()]

        #creates an empty list of chunks
        chunks: List[str] = []
        current = ""

        #go through each sentence, and if it less than 250, add the next sentence
        for s in sentences:
            if len(s) <= max_len:
                if not current:
                    current = s
                elif len(current) + 1 + len(s) <= max_len:
                    current += " " + s
                else: #if too long , make current the next sentence
                    chunks.append(current.strip())
                    current = s
            else:
                #if the sentence if too long, split it using, commas or semicolons
                parts = re.split(r"(?<=[,;:،])\s+", s)
                parts = [p.strip() for p in parts if p.strip()]

                #if its still too long, the cplit it at each 250 characters
                for part in parts:
                    if len(part) > max_len:
                        start = 0
                        while start < len(part):
                            piece = part[start:start + max_len].strip()
                            if piece:
                                if current:
                                    chunks.append(current.strip())
                                    current = ""
                                chunks.append(piece)
                            start += max_len
                    else:
                        if not current:
                            current = part
                        elif len(current) + 1 + len(part) <= max_len:
                            current += " " + part
                        else:
                            chunks.append(current.strip())
                            current = part

        if current.strip():
            chunks.append(current.strip())

        return chunks


    
    def load_subtitles(self) -> List[Dict[str, Any]]:
        if not os.path.exists(self.subtitles_json_path):
            raise FileNotFoundError(f"Subtitles JSON not found: {self.subtitles_json_path}")

        with open(self.subtitles_json_path, "r", encoding="utf-8") as f:
            data = json.load(f) #opens and reads the content

        slides: List[Dict[str, Any]] = [] #creates a list for the sentences to be spoken

        if isinstance(data, dict): #if dic is created, loop
            for key in sorted(data.keys(), key=lambda x: str(x)):
                slide = data[key]

                text_parts = []
                if isinstance(slide.get("items"), list): #if the slide has items
                    for item in slide["items"]:
                        sentence = self.clean_text(item.get("sentence", "")) #clean the text then store it 
                        if sentence:
                            text_parts.append(sentence)
                elif "text" in slide: #check for text also if items dont exisr
                    text_value = self.clean_text(slide.get("text", ""))
                    if text_value:
                        text_parts.append(text_value)
                elif "spoken_text" in slide: #also check for spoken text
                    text_value = self.clean_text(slide.get("spoken_text", ""))
                    if text_value:
                        text_parts.append(text_value)
                
                #store slide number and the text
                slides.append({
                    "slide_id": slide.get("slide_number", key),
                    "text": " ".join(text_parts).strip(),
                })

            return slides
        #if it was a list, do the same
        if isinstance(data, list):
            for idx, slide in enumerate(data, start=1):
                text_parts = []

                if isinstance(slide.get("items"), list):
                    for item in slide["items"]:
                        sentence = self.clean_text(item.get("sentence", ""))
                        if sentence:
                            text_parts.append(sentence)

                elif isinstance(slide.get("segments"), list):
                    for seg in slide["segments"]:
                        sentence = self.clean_text(seg.get("text", ""))
                        if sentence:
                            text_parts.append(sentence)

                elif "text" in slide:
                    text_value = self.clean_text(slide.get("text", ""))
                    if text_value:
                        text_parts.append(text_value)

                elif "spoken_text" in slide:
                    text_value = self.clean_text(slide.get("spoken_text", ""))
                    if text_value:
                        text_parts.append(text_value)

                slides.append({
                    "slide_id": slide.get("slide_number", slide.get("slide_id", idx)),
                    "text": " ".join(text_parts).strip(),
                })

            return slides

        raise ValueError("Unsupported subtitles JSON format")

    # -checks if voice exists
    def _voice_exists(self, voice_id: str) -> bool:
        url = f"https://api.elevenlabs.io/v1/voices/{voice_id}"
        resp = requests.get(url, headers=self.base_headers, timeout=60)
        return resp.status_code == 200 # if it returns 200, then it exists

    #checks if voice exist by name
    def _list_matching_cloned_voice_by_name(self, voice_name: str) -> Optional[str]: 
        ##prepares the API request to search for the name
        url = "https://api.elevenlabs.io/v2/voices"
        params = {
            "page_size": 100,
            "search": voice_name,
            "voice_type": "personal",
            "category": "cloned",
            "sort": "created_at_unix",
            "sort_direction": "desc",
            "include_total_count": "false",
        }

        resp = requests.get(url, headers=self.base_headers, params=params, timeout=60)
        resp.raise_for_status() #if error , then it failed
#the api request returns the payload which conatains a dictionary of all ids registered in the voice database
        payload = resp.json()
        for voice in payload.get("voices", []):
            name = (voice.get("name") or "").strip().lower()
            if name == voice_name.strip().lower():
                return voice.get("voice_id")

        return None

    #create a new voice clone
    def _create_instant_voice_clone(self, voice_name: str, audio_path: str) -> str:
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"Reference audio not found: {audio_path}")
        #prints the path , if it exists, and its size
        print(f"[Voice Debug] audio_path = {audio_path}")
        print(f"[Voice Debug] exists = {os.path.exists(audio_path)}")
        print(f"[Voice Debug] size = {os.path.getsize(audio_path)} bytes")
        #the url to request to add the voice
        url = "https://api.elevenlabs.io/v1/voices/add"
    
        mime_type = "audio/wav" #type of audio format (default)
        lower_name = audio_path.lower()
        if lower_name.endswith(".mp3"):
            mime_type = "audio/mpeg"
        elif lower_name.endswith(".m4a"):
            mime_type = "audio/mp4"
        elif lower_name.endswith(".ogg"):
            mime_type = "audio/ogg"
        elif lower_name.endswith(".flac"):
            mime_type = "audio/flac"
        #rb means binary mode
        with open(audio_path, "rb") as f:
            files = [
                ("files", (Path(audio_path).name, f, mime_type)),
            ]
            data = {
                "name": voice_name,
            }

            resp = requests.post(
                url,
                headers={"xi-api-key": self.api_key},
                data=data,
                files=files,
                timeout=300,
            )
        #check if it was not successful 
        if resp.status_code >= 400:
            raise RuntimeError(
                f"Voice clone creation failed ({resp.status_code}): {resp.text}"
            )
        #loads the list of voices and checks if its there
        payload = resp.json()
        voice_id = payload.get("voice_id")
        if not voice_id: #if it didnt then it prints this
            raise RuntimeError(f"Voice clone creation returned no voice_id: {payload}")

        return voice_id

    def ensure_voice_id(self) -> str:
        cache = self._load_voice_cache() #it loads the voice cache 
        audio_hash = None

        if self.ref_audio_path and os.path.exists(self.ref_audio_path):
            audio_hash = self._hash_file(self.ref_audio_path) #hashes the audio

        # 1) if it was in the cache, then get the already saved one
        if audio_hash and audio_hash in cache:
            cached_voice_id = cache[audio_hash].get("voice_id")
            if cached_voice_id and self._voice_exists(cached_voice_id):
                print(f"[Voice] Reusing cached voice_id: {cached_voice_id}")
                return cached_voice_id


        # 2) create new clone if reference audio exists
        if self.ref_audio_path and os.path.exists(self.ref_audio_path):
            print("[Voice] No reusable clone found. Creating new cloned voice...")
            new_voice_id = self._create_instant_voice_clone(
                voice_name=self.user_voice_name,
                audio_path=self.ref_audio_path,
            )
            print(f"[Voice] Created new cloned voice: {new_voice_id}")

            if audio_hash:
                cache[audio_hash] = {
                    "voice_id": new_voice_id,
                    "voice_name": self.user_voice_name,
                    "audio_path": self.ref_audio_path,
                }
                self._save_voice_cache(cache)

            return new_voice_id

        #  use a fallback if nothing exists
        if self.fallback_voice_id:
            print(f"[Voice] Using fallback voice_id: {self.fallback_voice_id}")
            return self.fallback_voice_id

        raise ValueError(
            "No reusable voice found and no reference audio available to create one."
        )

    # TTS generation
    # -------------------------------------------------
    def generate_chunk(self, chunk: str, out_file: str, voice_id: str) -> None:
        print("[ElevenLabs Request]")
        print("Voice ID:", voice_id)
        print("Model ID:", self.model_id)
        print("Output file:", out_file)
        print("Text:", chunk[:200])
        audio = self.client.text_to_speech.convert( #converts TTS by each function call
            voice_id=voice_id,
            text=chunk,
            model_id=self.model_id,
            output_format="mp3_44100_128",
        )
        save(audio, out_file)

    #main audio generation
    def generate_audio(self, text: str, out_path: str, slide_id: str, voice_id: str) -> None:
        text = self.clean_text(text) #first clean the text
        if not text.strip():
            print("[Skipped empty text]")
            return

        chunks = self.split_text(text) #then splot the text to sentences where each <250 word
        temp_files: List[str] = [] #test to store the temp mp3 

        for i, chunk in enumerate(chunks, start=1):
            print(f"  -> Chunk {i}/{len(chunks)}") 
            tmp_file = os.path.join(self.output_dir, f"tmp_slide_{slide_id}_{i}.mp3")
            self.generate_chunk(chunk, tmp_file, voice_id) #calls the TTS api
            temp_files.append(tmp_file) #appends audio to file

        combined = AudioSegment.empty() #create an audio segment to store the final adiop

        #combines the audio segments into one and adds a 1.5 second silence between
        for idx, f in enumerate(temp_files):
            audio_seg = AudioSegment.from_file(f)
            combined += audio_seg
            if idx < len(temp_files) - 1:
                combined += AudioSegment.silent(duration=self.silence_ms)

        combined.export(out_path, format="wav")

        for f in temp_files:
            if os.path.exists(f):
                os.remove(f)

    #this function is process one slide then calls the generate audio function
    async def process_slide(self, slide: Dict[str, Any], voice_id: str) -> None:
        slide_id = slide["slide_id"]
        print(f"\n[Slide {slide_id}] Generating audio...")

        out_path = os.path.join(self.output_dir, f"slide_{slide_id}.wav")
        self.generate_audio(slide["text"], out_path, str(slide_id), voice_id)

        print(f"[Saved] {out_path}")

    #controls the complete speech generation
    async def run_async(self, limit_slides: Optional[int] = None) -> None:
        slides = self.load_subtitles() #loads all subtitles

        if limit_slides is not None:
            slides = slides[:limit_slides]

        #ensures audio exists
        print(f"[Speech] Processing {len(slides)} slides...")
        voice_id = self.ensure_voice_id()
        print(f"[Speech] Using voice_id: {voice_id}")

        #processes the slide by slide and does TTS
        for slide in slides:
            await self.process_slide(slide, voice_id)

        print("[DONE]")

    #runs the main speech running pipeline
    def run(
        self,
        limit_slides: Optional[int] = None,
        subtitles_json_path: Optional[str] = None,
        ref_audio_path: Optional[str] = None,
        user_voice_name: Optional[str] = None,
    ) -> str:
        if subtitles_json_path:
            self.subtitles_json_path = subtitles_json_path
        if ref_audio_path:
            self.ref_audio_path = ref_audio_path
        if user_voice_name:
            self.user_voice_name = user_voice_name

        asyncio.run(self.run_async(limit_slides))
        return self.output_dir


if __name__ == "__main__":
    agent = SpeechAgent(
        subtitles_json_path="Data/intermediate/final_subtitles.json",
        output_dir="Data/intermediate/speech_output",
        ref_audio_path="Data/input/ref_clean.wav",
        api_key="sk_0f4715ea095118af6803da9ad114513bc59d53392bea09d1",
        user_voice_name="talexa_user_voice",
        voice_cache_path="Data/intermediate/voice_cache.json",
        fallback_voice_id=None,
        model_id="eleven_multilingual_v2",
    )
    agent.run()
