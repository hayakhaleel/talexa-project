import argparse
import glob
import hashlib
import json
import mimetypes
import os
import re
import subprocess
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from PIL import Image, ImageOps

class TalkingHeadApiAgent:
    def __init__(
        self,
        source_image: str,
        audio_dir: str,
        output_dir: str = "Data/intermediate/talking_head_api",
        api_key: Optional[str] = None,
        ffmpeg_binary: str = "ffmpeg",
        title: str = "Talking Head API Video",
        poll_interval_seconds: int = 10,
        timeout_seconds: int = 1800,
        use_avatar_iv_model: bool = True,
        max_image_side: int = 1536,
        jpeg_quality: int = 95,
    ):
        self.source_image = os.path.abspath(source_image)
        self.audio_dir = os.path.abspath(audio_dir)
        self.output_dir = os.path.abspath(output_dir)
        self.api_key = api_key or os.getenv("HEYGEN_API_KEY")
        self.ffmpeg_binary = ffmpeg_binary
        self.title = title
        self.poll_interval_seconds = poll_interval_seconds
        self.timeout_seconds = timeout_seconds
        self.use_avatar_iv_model = use_avatar_iv_model
        self.max_image_side = max_image_side
        self.jpeg_quality = jpeg_quality

        os.makedirs(self.output_dir, exist_ok=True)

        self.merged_wav_path = os.path.join(self.output_dir, "merged_slides.wav")
        self.upload_mp3_path = os.path.join(self.output_dir, "merged_slides_upload.mp3")
        self.prepared_image_path = os.path.join(self.output_dir, "prepared_source.jpg")
        self.output_video_path = os.path.join(self.output_dir, "talking_head.mp4")
        self.cache_path = os.path.join(self.output_dir, "avatar_cache.json")
        self.debug_dir = os.path.join(self.output_dir, "debug")
        os.makedirs(self.debug_dir, exist_ok=True)

    def _require_api_key(self) -> None:
        if not self.api_key:
            raise ValueError(
                "HeyGen API key is required. Pass api_key=... or set HEYGEN_API_KEY."
            )

    def _require_inputs(self) -> None:
        if not os.path.exists(self.source_image):
            raise FileNotFoundError(f"Source image not found: {self.source_image}")
        if not os.path.isdir(self.audio_dir):
            raise FileNotFoundError(f"Audio directory not found: {self.audio_dir}")

    def _run_ffmpeg(self, args: List[str]) -> None:
        cmd = [self.ffmpeg_binary, *args]
        print("[FFmpeg]", " ".join(cmd))
        subprocess.run(cmd, check=True)

    def _natural_sort_key(self, path: str) -> List[Any]:
        parts = re.split(r"(\d+)", Path(path).name.lower())
        key: List[Any] = []
        for part in parts:
            key.append(int(part) if part.isdigit() else part)
        return key

    def _save_debug_json(self, filename: str, payload: Any) -> None:
        path = os.path.join(self.debug_dir, filename)
        try:
            with open(path, "w", encoding="utf-8") as handle:
                json.dump(payload, handle, indent=2, ensure_ascii=False)
        except Exception:
            pass

    def _hash_file(self, file_path: str) -> str:
        hasher = hashlib.sha256()
        with open(file_path, "rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                hasher.update(chunk)
        return hasher.hexdigest()

    def _load_cache(self) -> Dict[str, Any]:
        if not os.path.exists(self.cache_path):
            return {}
        try:
            with open(self.cache_path, "r", encoding="utf-8") as handle:
                data = json.load(handle)
            return data if isinstance(data, dict) else {}
        except Exception:
            return {}

    def _save_cache(self, cache: Dict[str, Any]) -> None:
        with open(self.cache_path, "w", encoding="utf-8") as handle:
            json.dump(cache, handle, indent=2, ensure_ascii=False)

    def get_audio_files(self) -> List[str]:
        candidates = []
        for entry in os.listdir(self.audio_dir):
            if entry.lower().endswith(".wav"):
                candidates.append(os.path.join(self.audio_dir, entry))

        candidates.sort(key=self._natural_sort_key)

        if not candidates:
            raise FileNotFoundError(f"No .wav files found in: {self.audio_dir}")

        return candidates

    def merge_audio_with_silence(self, audio_files: Iterable[str]) -> str:
        audio_files = list(audio_files)
        if not audio_files:
            raise ValueError("No audio files were provided for merging.")

        with tempfile.TemporaryDirectory(prefix="talking_head_api_") as temp_dir:
            normalized_files = []

            for index, audio_path in enumerate(audio_files, start=1):
                normalized_path = os.path.join(temp_dir, f"audio_{index:03d}.wav")
                self._run_ffmpeg(
                    [
                        "-y",
                        "-i",
                        audio_path,
                        "-ac",
                        "1",
                        "-ar",
                        "44100",
                        "-c:a",
                        "pcm_s16le",
                        normalized_path,
                    ]
                )
                normalized_files.append(normalized_path)

            silence_path = os.path.join(temp_dir, "silence_1s.wav")
            self._run_ffmpeg(
                [
                    "-y",
                    "-f",
                    "lavfi",
                    "-i",
                    "anullsrc=r=44100:cl=mono",
                    "-t",
                    "1",
                    "-c:a",
                    "pcm_s16le",
                    silence_path,
                ]
            )

            concat_list_path = os.path.join(temp_dir, "concat_list.txt")
            with open(concat_list_path, "w", encoding="utf-8") as handle:
                for index, normalized_path in enumerate(normalized_files):
                    normalized_for_ffmpeg = normalized_path.replace("\\", "/").replace("'", "'\\''")
                    handle.write(f"file '{normalized_for_ffmpeg}'\n")
                    if index < len(normalized_files) - 1:
                        silence_for_ffmpeg = silence_path.replace("\\", "/").replace("'", "'\\''")
                        handle.write(f"file '{silence_for_ffmpeg}'\n")

            self._run_ffmpeg(
                [
                    "-y",
                    "-f",
                    "concat",
                    "-safe",
                    "0",
                    "-i",
                    concat_list_path,
                    "-c:a",
                    "pcm_s16le",
                    self.merged_wav_path,
                ]
            )

        print(f"[TalkingHeadAPI] Merged WAV saved to: {self.merged_wav_path}")
        return self.merged_wav_path

    def convert_wav_to_mp3(self, wav_path: str) -> str:
        self._run_ffmpeg(
            [
                "-y",
                "-i",
                wav_path,
                "-codec:a",
                "libmp3lame",
                "-q:a",
                "2",
                self.upload_mp3_path,
            ]
        )
        print(f"[TalkingHeadAPI] Upload MP3 saved to: {self.upload_mp3_path}")
        return self.upload_mp3_path

    def prepare_image(self) -> Tuple[str, str]:
        image = Image.open(self.source_image)
        image = ImageOps.exif_transpose(image).convert("RGB")

        width, height = image.size
        max_side = max(width, height)
        if max_side > self.max_image_side:
            scale = self.max_image_side / float(max_side)
            new_size = (max(1, int(width * scale)), max(1, int(height * scale)))
            image = image.resize(new_size, Image.LANCZOS)

        image.save(self.prepared_image_path, format="JPEG", quality=self.jpeg_quality, optimize=True)

        image_hash = self._hash_file(self.prepared_image_path)
        print(f"[TalkingHeadAPI] Prepared image saved to: {self.prepared_image_path}")
        print(f"[TalkingHeadAPI] Prepared image hash: {image_hash}")
        return self.prepared_image_path, image_hash

    def _http_json(
        self,
        method: str,
        url: str,
        *,
        headers: Optional[Dict[str, str]] = None,
        body: Optional[bytes] = None,
    ) -> Dict[str, Any]:
        request = urllib.request.Request(url, data=body, method=method)
        request.add_header("X-Api-Key", self.api_key or "")
        for key, value in (headers or {}).items():
            request.add_header(key, value)

        try:
            with urllib.request.urlopen(request, timeout=120) as response:
                content = response.read().decode("utf-8")
                status_code = getattr(response, "status", 200)
        except urllib.error.HTTPError as exc:
            details = exc.read().decode("utf-8", errors="replace")
            try:
                payload = json.loads(details)
            except Exception:
                payload = {"raw_text": details}
            self._save_debug_json(
                f"http_error_{method.lower()}_{Path(urllib.parse.urlparse(url).path).name or 'root'}.json",
                {
                    "url": url,
                    "status_code": exc.code,
                    "headers": headers or {},
                    "response": payload,
                },
            )
            raise RuntimeError(
                f"HeyGen API request failed ({exc.code}) for {url}: {details}"
            ) from exc

        if not content:
            return {}

        try:
            payload = json.loads(content)
        except Exception:
            payload = {"raw_text": content}

        self._save_debug_json(
            f"http_{method.lower()}_{Path(urllib.parse.urlparse(url).path).name or 'root'}.json",
            {
                "url": url,
                "status_code": status_code,
                "headers": headers or {},
                "response": payload,
            },
        )
        return payload

    def _extract_first_value(
        self,
        data: Any,
        *,
        preferred_keys: Iterable[str],
    ) -> Optional[Any]:
        preferred_keys = tuple(preferred_keys)

        def walk(node: Any) -> Optional[Any]:
            if isinstance(node, dict):
                for key in preferred_keys:
                    value = node.get(key)
                    if value not in (None, "", [], {}):
                        return value
                for value in node.values():
                    found = walk(value)
                    if found not in (None, "", [], {}):
                        return found
            elif isinstance(node, list):
                for item in node:
                    found = walk(item)
                    if found not in (None, "", [], {}):
                        return found
            return None

        return walk(data)

    def _extract_video_url(self, data: Any) -> Optional[str]:
        candidate = self._extract_first_value(
            data,
            preferred_keys=("video_url", "url", "video_url_without_captions"),
        )
        return candidate if isinstance(candidate, str) and candidate.startswith("http") else None

    def upload_asset_raw(self, file_path: str, content_type: str) -> Dict[str, Any]:
        with open(file_path, "rb") as handle:
            body = handle.read()

        print(f"[TalkingHeadAPI] Uploading asset: {file_path} ({content_type})")
        return self._http_json(
            "POST",
            "https://upload.heygen.com/v1/asset",
            headers={"Content-Type": content_type},
            body=body,
        )

    def extract_image_key(self, payload: Dict[str, Any]) -> str:
        data = payload.get("data", {}) if isinstance(payload, dict) else {}
        image_key = data.get("image_key") or data.get("key") or payload.get("image_key")
        if not image_key:
            raise RuntimeError(f"Could not find image_key in upload response: {payload}")
        return str(image_key)

    def extract_audio_asset_id(self, payload: Dict[str, Any]) -> str:
        data = payload.get("data", {}) if isinstance(payload, dict) else {}
        audio_asset_id = (
            data.get("asset_id")
            or data.get("audio_asset_id")
            or data.get("id")
            or payload.get("asset_id")
            or payload.get("id")
        )
        if not audio_asset_id:
            raise RuntimeError(f"Could not find audio asset id in upload response: {payload}")
        return str(audio_asset_id)

    def create_photo_avatar_group(self, image_key: str) -> str:
        payload = {"name": self.title, "image_key": image_key}
        response = self._http_json(
            "POST",
            "https://api.heygen.com/v2/photo_avatar/avatar_group/create",
            headers={"Content-Type": "application/json"},
            body=json.dumps(payload).encode("utf-8"),
        )

        group_id = (
            self._extract_first_value(response, preferred_keys=("group_id", "id"))
        )
        if not group_id:
            raise RuntimeError(f"Could not find group id in response: {response}")

        return str(group_id)

    def list_avatars_in_group(self, group_id: str) -> List[Dict[str, Any]]:
        response = self._http_json(
            "GET",
            f"https://api.heygen.com/v2/avatar_group/{group_id}/avatars",
        )

        data = response.get("data")
        if isinstance(data, dict):
            avatars = data.get("avatar_list") or data.get("avatars") or data.get("items") or []
        elif isinstance(data, list):
            avatars = data
        else:
            avatars = response.get("avatars") or []

        return [avatar for avatar in avatars if isinstance(avatar, dict)]

    def wait_for_avatar_id(self, group_id: str) -> str:
        started = time.time()
        ready_statuses = {"completed", "ready", "trained", "success", "succeeded"}

        while True:
            avatars = self.list_avatars_in_group(group_id)

            if avatars:
                avatar = avatars[0]
                avatar_id = avatar.get("id") or avatar.get("avatar_id") or avatar.get("talking_photo_id")
                status = str(avatar.get("status", "")).lower().strip()

                print(f"[TalkingHeadAPI] Avatar status: {status or 'unknown'} | avatar_id: {avatar_id}")

                if avatar_id and (status in ready_statuses or status not in {"pending", "processing", "uploaded"}):
                    return str(avatar_id)

            if time.time() - started > self.timeout_seconds:
                raise TimeoutError(
                    f"Timed out after {self.timeout_seconds} seconds waiting for avatar."
                )

            time.sleep(self.poll_interval_seconds)

    def resolve_avatar_id(self) -> str:
        prepared_image_path, image_hash = self.prepare_image()

        cache = self._load_cache()
        cached = cache.get(image_hash)
        if isinstance(cached, dict) and cached.get("avatar_id"):
            print(f"[TalkingHeadAPI] Reusing cached avatar_id: {cached['avatar_id']}")
            return str(cached["avatar_id"])

        image_upload = self.upload_asset_raw(prepared_image_path, "image/jpeg")
        image_key = self.extract_image_key(image_upload)

        group_id = self.create_photo_avatar_group(image_key)
        avatar_id = self.wait_for_avatar_id(group_id)

        cache[image_hash] = {
            "avatar_id": avatar_id,
            "group_id": group_id,
            "source_image": self.source_image,
            "prepared_image_path": prepared_image_path,
            "created_at": int(time.time()),
        }
        self._save_cache(cache)

        return avatar_id


    def create_video(self, avatar_id: str, audio_asset_id: str) -> str:
        payload = {
            "title": self.title,
            "video_inputs": [
                {
                    "character": {
                        "type": "talking_photo",
                        "talking_photo_id": avatar_id,
                    },
                    "voice": {
                        "type": "audio",
                        "audio_asset_id": audio_asset_id,
                    },
                }
            ],
        }

        if self.use_avatar_iv_model:
            payload["use_avatar_iv_model"] = True

        print(f"[TalkingHeadAPI] Creating video with avatar_id={avatar_id} audio_asset_id={audio_asset_id}")
        response = self._http_json(
            "POST",
            "https://api.heygen.com/v2/video/generate",
            headers={"Content-Type": "application/json"},
            body=json.dumps(payload).encode("utf-8"),
        )

        video_id = self._extract_first_value(
            response,
            preferred_keys=("video_id", "id"),
        )
        if not video_id:
            raise RuntimeError(f"Could not find video id in response: {response}")

        return str(video_id)

    def get_video_status(self, video_id: str) -> Dict[str, Any]:
        query = urllib.parse.urlencode({"video_id": video_id})
        url = f"https://api.heygen.com/v1/video_status.get?{query}"
        return self._http_json("GET", url)

    def wait_for_video(self, video_id: str) -> str:
        started = time.time()

        while True:
            response = self.get_video_status(video_id)
            status = str(
                self._extract_first_value(
                    response,
                    preferred_keys=("status", "video_status"),
                )
                or ""
            ).lower()

            print(f"[TalkingHeadAPI] Video status: {status or 'unknown'}")

            if status == "completed":
                video_url = self._extract_video_url(response)
                if not video_url:
                    raise RuntimeError(
                        f"Video completed but no downloadable URL was found: {response}"
                    )
                return video_url

            if status == "failed":
                raise RuntimeError(f"HeyGen video generation failed: {response}")

            if time.time() - started > self.timeout_seconds:
                raise TimeoutError(
                    f"Timed out after {self.timeout_seconds} seconds waiting for video."
                )

            time.sleep(self.poll_interval_seconds)

    def download_file(self, url: str, output_path: str) -> str:
        print(f"[TalkingHeadAPI] Downloading video from: {url}")
        request = urllib.request.Request(url, method="GET")

        try:
            with urllib.request.urlopen(request, timeout=300) as response:
                with open(output_path, "wb") as handle:
                    handle.write(response.read())
        except urllib.error.HTTPError as exc:
            details = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(
                f"Could not download generated video ({exc.code}): {details}"
            ) from exc

        print(f"[TalkingHeadAPI] Final MP4 saved to: {output_path}")
        return output_path

    def run(self) -> Dict[str, str]:
        self._require_api_key()
        self._require_inputs()

        audio_files = self.get_audio_files()
        print(f"[TalkingHeadAPI] Found {len(audio_files)} wav file(s).")

        merged_wav_path = self.merge_audio_with_silence(audio_files)
        upload_mp3_path = self.convert_wav_to_mp3(merged_wav_path)

        avatar_id = self.resolve_avatar_id()

        audio_upload = self.upload_asset_raw(upload_mp3_path, "audio/mpeg")
        audio_asset_id = self.extract_audio_asset_id(audio_upload)

        video_id = self.create_video(avatar_id, audio_asset_id)
        video_url = self.wait_for_video(video_id)
        final_video_path = self.download_file(video_url, self.output_video_path)

        result = {
            "merged_wav_path": merged_wav_path,
            "upload_mp3_path": upload_mp3_path,
            "final_video_path": final_video_path,
            "video_id": video_id,
            "video_url": video_url,
            "avatar_id": avatar_id,
            "audio_asset_id": audio_asset_id,
        }

        print("[TalkingHeadAPI] Done.")
        print(json.dumps(result, indent=2))
        return result


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Merge slide audio, create/reuse a photo avatar, and generate a HeyGen talking head MP4."
    )
    parser.add_argument("--image", required=True, help="Path to the portrait image.")
    parser.add_argument(
        "--audio-dir",
        required=True,
        help="Folder containing slide audio WAV files such as slide_1.wav, slide_2.wav.",
    )
    parser.add_argument(
        "--output-dir",
        default="Data/intermediate/talking_head_api",
        help="Where merged audio and final video should be saved.",
    )
    parser.add_argument(
        "--api-key",
        default=None,
        help="HeyGen API key. If omitted, HEYGEN_API_KEY is used.",
    )
    parser.add_argument(
        "--title",
        default="Talking Head API Video",
        help="Display title for the generated HeyGen video.",
    )
    parser.add_argument(
        "--poll-interval",
        type=int,
        default=10,
        help="Seconds between status checks.",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=1800,
        help="Maximum number of seconds to wait for rendering.",
    )
    parser.add_argument(
        "--ffmpeg-binary",
        default="ffmpeg",
        help="Path or command name for ffmpeg.",
    )
    return parser


if __name__ == "__main__":
    args = build_arg_parser().parse_args()

    agent = TalkingHeadApiAgent(
        source_image=args.image,
        audio_dir=args.audio_dir,
        output_dir=args.output_dir,
        api_key=args.api_key,
        ffmpeg_binary=args.ffmpeg_binary,
        title=args.title,
        poll_interval_seconds=args.poll_interval,
        timeout_seconds=args.timeout,
    )
    agent.run() 
