from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any


MODEL_DIR = Path(__file__).resolve().parents[1] / "models" / "classifier_model"
CLASS_NAMES_PATH = MODEL_DIR / "class_names.json"
CHECKPOINT_PATH = MODEL_DIR / "resnet18_valid_invalid_checkpoint.pth"


def _strip_state_dict_prefix(state_dict: dict[str, Any]) -> dict[str, Any]:
    cleaned_state_dict: dict[str, Any] = {}

    for key, value in state_dict.items():
        cleaned_key = key
        for prefix in ("module.", "model."):
            if cleaned_key.startswith(prefix):
                cleaned_key = cleaned_key[len(prefix):]
        cleaned_state_dict[cleaned_key] = value

    return cleaned_state_dict


class PortraitClassifier:
    def __init__(self) -> None:
        try:
            import torch
            from PIL import Image
            from torchvision import models, transforms
        except ImportError as exc:
            raise RuntimeError(
                "Portrait classifier dependencies are missing. "
                "Please install torch, torchvision, and Pillow."
            ) from exc

        if not CLASS_NAMES_PATH.exists():
            raise FileNotFoundError(f"Classifier labels not found: {CLASS_NAMES_PATH}")
        if not CHECKPOINT_PATH.exists():
            raise FileNotFoundError(f"Classifier checkpoint not found: {CHECKPOINT_PATH}")

        self._torch = torch
        self._image_module = Image
        self._transform = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ]
        )
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        with CLASS_NAMES_PATH.open("r", encoding="utf-8") as file:
            self.class_names = json.load(file)

        if not isinstance(self.class_names, list) or not self.class_names:
            raise ValueError("Classifier class_names.json must contain a non-empty list.")

        model = models.resnet18(weights=None)
        model.fc = torch.nn.Linear(model.fc.in_features, len(self.class_names))

        checkpoint = torch.load(CHECKPOINT_PATH, map_location=self.device)
        if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        else:
            state_dict = checkpoint

        if not isinstance(state_dict, dict):
            raise ValueError("Unsupported classifier checkpoint format.")

        model.load_state_dict(_strip_state_dict_prefix(state_dict), strict=False)
        model.to(self.device)
        model.eval()
        self.model = model

    def classify(self, image_path: str | Path) -> dict[str, Any]:
        image_path = Path(image_path).expanduser().resolve()
        image = self._image_module.open(image_path).convert("RGB")
        input_tensor = self._transform(image).unsqueeze(0).to(self.device)

        with self._torch.inference_mode():
            logits = self.model(input_tensor)
            probabilities = self._torch.softmax(logits, dim=1)[0]
            predicted_index = int(self._torch.argmax(probabilities).item())

        predicted_label = str(self.class_names[predicted_index])
        confidence = float(probabilities[predicted_index].item())
        is_valid = predicted_label.strip().lower() == "valid"

        return {
            "label": predicted_label,
            "confidence": confidence,
            "is_valid": is_valid,
        }


@lru_cache(maxsize=1)
def get_portrait_classifier() -> PortraitClassifier:
    return PortraitClassifier()
