from __future__ import annotations

import importlib.util
from pathlib import Path
import subprocess
import sys


MODEL_NAME = "facebook/nllb-200-distilled-600M"
REQUIRED_PACKAGES = ("transformers", "sentencepiece", "accelerate", "torch")


def pip_command() -> list[str]:
    if importlib.util.find_spec("pip") is not None:
        return [sys.executable, "-m", "pip", "install"]

    venv_dir = Path(sys.executable).resolve().parents[1]
    pyvenv_cfg = venv_dir / "pyvenv.cfg"
    if pyvenv_cfg.exists():
        for line in pyvenv_cfg.read_text(encoding="utf-8").splitlines():
            key, _, value = line.partition("=")
            if key.strip() == "executable" and value.strip():
                base_python = Path(value.strip())
                if base_python.exists():
                    return [
                        str(base_python),
                        "-m",
                        "pip",
                        "--python",
                        sys.executable,
                        "install",
                    ]

    raise RuntimeError(
        "pip is not available in this environment, and a base Python with pip "
        "could not be found from pyvenv.cfg."
    )


def ensure_requirements() -> None:
    missing_packages = [
        package
        for package in REQUIRED_PACKAGES
        if importlib.util.find_spec(package) is None
    ]

    if not missing_packages:
        print("Required packages are already installed.")
        return

    print("Installing missing packages:", ", ".join(missing_packages))
    subprocess.check_call([*pip_command(), *missing_packages])


def main() -> None:
    ensure_requirements()

    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    print(f"Downloading/caching translation model: {MODEL_NAME}")
    AutoTokenizer.from_pretrained(MODEL_NAME)
    AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)
    print("NLLB translation model is ready.")


if __name__ == "__main__":
    main()
