#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Prepare the local runtime environment for faster-whisper + ONNX M2M100.

User flow:
    1. Edit setup_env.json.
    2. Run: python setup_env.py
    3. Run: python fasterwhisper_m2m100_server.py

This script downloads only the assets selected in setup_env.json. If the target
files already exist, the corresponding download/export step is skipped.
"""

from __future__ import annotations

import gc
import json
import sys
import time
from pathlib import Path
from typing import Any, Mapping

BASE_DIR = Path(__file__).resolve().parent
CONFIG_FILE = BASE_DIR / "setup_env.json"

DEFAULT_CONFIG: dict[str, Any] = {
    "setup": {
        "models_dir": "models",
        "hf_cache_dir": "hf_cache",
        "download_faster_whisper": True,
        "download_m2m100": True,
    },
    "transcribe": {
        "FW_MODEL_NAME": "small",
        "FW_COMPUTE_TYPE": "auto",
        "FW_DEVICE": "cpu",
    },
    "translation": {
        "M2M_MODEL_SIZE": "418M",
        "M2M_ORT_PROVIDER": "CPUExecutionProvider",
        "MAX_INPUT_LENGTH": 512,
        "MAX_OUTPUT_LENGTH": 256,
        "NUM_BEAMS": 2,
    },
    "server": {
        "HOST": "0.0.0.0",
        "HTTP_PORT": 9000,
        "HTTPS_PORT": 9443,
        "CERT_FILE": "server.crt",
        "KEY_FILE": "server.key",
    },
}


def deep_merge(base: Mapping[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    """Return a recursive merge of two dictionaries."""
    merged: dict[str, Any] = dict(base)
    for key, value in override.items():
        if isinstance(value, Mapping) and isinstance(merged.get(key), Mapping):
            merged[key] = deep_merge(merged[key], value)  # type: ignore[arg-type]
        else:
            merged[key] = value
    return merged


def load_config(config_path: Path | str = CONFIG_FILE) -> dict[str, Any]:
    """Load setup_env.json and apply defaults for missing keys."""
    path = Path(config_path)
    if not path.is_absolute():
        path = BASE_DIR / path

    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found: {path}")

    with path.open("r", encoding="utf-8") as f:
        loaded = json.load(f)

    if not isinstance(loaded, Mapping):
        raise ValueError(f"Configuration root must be a JSON object: {path}")

    return deep_merge(DEFAULT_CONFIG, loaded)


def resolve_project_path(path_value: str | Path) -> Path:
    """Resolve a path relative to the project directory."""
    path = Path(path_value)
    return path if path.is_absolute() else BASE_DIR / path


def get_models_dir(config: Mapping[str, Any]) -> Path:
    """Return the configured models directory."""
    setup = config.get("setup", {})
    models_dir = setup.get("models_dir", "models") if isinstance(setup, Mapping) else "models"
    return resolve_project_path(models_dir)


def get_hf_cache_dir(config: Mapping[str, Any]) -> Path:
    """Return the configured Hugging Face cache directory."""
    setup = config.get("setup", {})
    hf_cache_dir = setup.get("hf_cache_dir", "hf_cache") if isinstance(setup, Mapping) else "hf_cache"
    return resolve_project_path(hf_cache_dir)


def get_faster_whisper_model_name(config: Mapping[str, Any]) -> str:
    """Return the configured faster-whisper model name or local path."""
    transcribe = config.get("transcribe", {})
    if not isinstance(transcribe, Mapping):
        return str(DEFAULT_CONFIG["transcribe"]["FW_MODEL_NAME"])
    return str(transcribe.get("FW_MODEL_NAME", DEFAULT_CONFIG["transcribe"]["FW_MODEL_NAME"])).strip()


def get_m2m100_model_size(config: Mapping[str, Any]) -> str:
    """Return the configured M2M100 size as either 418M or 1.2B."""
    translation = config.get("translation", {})
    raw_size = "418M"
    if isinstance(translation, Mapping):
        raw_size = str(translation.get("M2M_MODEL_SIZE", "418M")).strip()

    normalized = raw_size.upper().replace("_", "").replace("-", "")
    if normalized in {"1.2B", "12B", "1200M"}:
        return "1.2B"
    return "418M"


def get_m2m100_model_dir(config: Mapping[str, Any]) -> Path:
    """Return the ONNX M2M100 output directory used by the runtime."""
    models_dir = get_models_dir(config)
    if get_m2m100_model_size(config) == "1.2B":
        return models_dir / "onnx-m2m100-1.2B"
    return models_dir / "onnx-m2m100"


def bool_from_config(config: Mapping[str, Any], key: str, default: bool = True) -> bool:
    """Read a boolean from the setup section."""
    setup = config.get("setup", {})
    value = setup.get(key, default) if isinstance(setup, Mapping) else default
    return bool(value)


def has_required_files(directory: Path, filenames: list[str]) -> bool:
    """Return True if all required files exist in a directory."""
    return directory.exists() and all((directory / name).is_file() for name in filenames)


def is_m2m100_export_complete(model_dir: Path) -> bool:
    """Return True if the ONNX M2M100 export appears complete."""
    required = [
        "config.json",
        "encoder_model.onnx",
        "decoder_model.onnx",
        "decoder_with_past_model.onnx",
        "sentencepiece.bpe.model",
        "tokenizer_config.json",
        "special_tokens_map.json",
    ]
    return has_required_files(model_dir, required)


def candidate_faster_whisper_dirs(models_dir: Path, model_name: str) -> list[Path]:
    """Return common faster-whisper cache/export directories for a model name."""
    clean_name = model_name.replace("/", "--")
    return [
        models_dir / model_name,
        models_dir / f"faster-whisper-{model_name}",
        models_dir / f"models--Systran--faster-whisper-{model_name}",
        models_dir / clean_name,
    ]


def looks_like_faster_whisper_model(path: Path) -> bool:
    """Return True if a directory looks like a faster-whisper CTranslate2 model."""
    return path.is_dir() and (path / "model.bin").is_file() and (path / "config.json").is_file()


def is_faster_whisper_available(models_dir: Path, model_name: str) -> bool:
    """Return True if the selected faster-whisper model already exists locally."""
    model_path = Path(model_name)
    if model_path.exists():
        return True

    for candidate in candidate_faster_whisper_dirs(models_dir, model_name):
        if looks_like_faster_whisper_model(candidate):
            return True

    if models_dir.exists():
        for candidate in models_dir.rglob("model.bin"):
            parent = candidate.parent
            parent_text = str(parent).lower()
            if "faster-whisper" in parent_text and model_name.lower() in parent_text:
                if (parent / "config.json").is_file():
                    return True

    return False


def download_faster_whisper(config: Mapping[str, Any]) -> None:
    """Download the configured faster-whisper model unless it already exists."""
    models_dir = get_models_dir(config)
    model_name = get_faster_whisper_model_name(config)
    models_dir.mkdir(parents=True, exist_ok=True)

    if is_faster_whisper_available(models_dir, model_name):
        print(f"[skip] faster-whisper model already exists: {model_name}")
        return

    print(f"[download] faster-whisper model: {model_name}")
    from faster_whisper import WhisperModel

    # Keep the original project behavior: WhisperModel downloads to ./models.
    model = WhisperModel(model_name, download_root=str(models_dir))
    del model
    gc.collect()
    print(f"[done] faster-whisper model downloaded: {model_name}")


def export_m2m100(config: Mapping[str, Any]) -> None:
    """Export the configured M2M100 model to ONNX unless it already exists."""
    model_size = get_m2m100_model_size(config)
    model_id = "facebook/m2m100_1.2B" if model_size == "1.2B" else "facebook/m2m100_418M"
    save_dir = get_m2m100_model_dir(config)
    hf_cache = get_hf_cache_dir(config)

    if is_m2m100_export_complete(save_dir):
        print(f"[skip] ONNX M2M100 model already exists: {save_dir}")
        return

    save_dir.mkdir(parents=True, exist_ok=True)
    hf_cache.mkdir(parents=True, exist_ok=True)

    print(f"[export] ONNX M2M100 model: {model_id}")
    print(f"[path] save_dir={save_dir}")
    print(f"[path] hf_cache={hf_cache}")

    from transformers import M2M100Tokenizer
    from optimum.onnxruntime import ORTModelForSeq2SeqLM

    tokenizer = M2M100Tokenizer.from_pretrained(model_id, cache_dir=str(hf_cache))
    onnx_model = ORTModelForSeq2SeqLM.from_pretrained(
        model_id,
        export=True,
        cache_dir=str(hf_cache),
    )
    onnx_model.save_pretrained(save_dir)
    tokenizer.save_pretrained(save_dir)

    del onnx_model
    del tokenizer
    gc.collect()
    time.sleep(1)
    print(f"[done] ONNX M2M100 model and tokenizer exported: {save_dir}")


def main() -> int:
    """Run all configured setup steps."""
    try:
        config = load_config()
        print(f"[config] {CONFIG_FILE}")

        if bool_from_config(config, "download_faster_whisper", True):
            download_faster_whisper(config)
        else:
            print("[skip] faster-whisper download is disabled in setup_env.json")

        if bool_from_config(config, "download_m2m100", True):
            export_m2m100(config)
        else:
            print("[skip] M2M100 download/export is disabled in setup_env.json")

        print("[ready] Environment setup completed.")
        return 0
    except Exception as exc:
        print(f"[error] Setup failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
