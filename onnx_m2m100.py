#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys

import onnxruntime as ort
from optimum.onnxruntime import ORTModelForSeq2SeqLM
from transformers import M2M100Tokenizer

from setup_env import load_config, get_m2m100_model_dir, get_m2m100_model_size

CONFIG = load_config()
TRANSLATION_CONFIG = CONFIG.get("translation", {})

MODEL_SIZE = get_m2m100_model_size(CONFIG)
MODEL_DIR = get_m2m100_model_dir(CONFIG)
REQUESTED_PROVIDER = str(TRANSLATION_CONFIG.get("M2M_ORT_PROVIDER", "CPUExecutionProvider"))
MAX_INPUT_LENGTH = int(TRANSLATION_CONFIG.get("MAX_INPUT_LENGTH", 512))
MAX_OUTPUT_LENGTH = int(TRANSLATION_CONFIG.get("MAX_OUTPUT_LENGTH", 256))
NUM_BEAMS = int(TRANSLATION_CONFIG.get("NUM_BEAMS", 2))

print(f"Loading ONNX M2M100 from: {MODEL_DIR}")

if not MODEL_DIR.exists():
    raise RuntimeError(
        f"ONNX model directory does not exist: {MODEL_DIR}. "
        "Run 'python setup_env.py' first."
    )

available_providers = ort.get_available_providers()
print(f"ONNX Runtime available providers: {available_providers}")

provider = REQUESTED_PROVIDER if REQUESTED_PROVIDER in available_providers else "CPUExecutionProvider"
print(f"Using ONNX Runtime provider: {provider}")

tokenizer = M2M100Tokenizer.from_pretrained(str(MODEL_DIR), local_files_only=True)
model = ORTModelForSeq2SeqLM.from_pretrained(
    str(MODEL_DIR),
    local_files_only=True,
    provider=provider,
    use_merged=False,
    use_io_binding=False,
)


def m2m100(from_lang, to_lang, transcribed_text):

    iso_map = {
    # Core / East Asia
    "en": "en",
    "zh-cn": "zh",
    "zh-tw": "zh",
    "zh": "zh",
    "ko": "ko",
    "ja": "ja",

    # Major European languages
    "de": "de",
    "fr": "fr",
    "es": "es",
    "pt": "pt",
    "nl": "nl",
    "it": "it",
    "tr": "tr",
    "pl": "pl",
    "sv": "sv",
    "no": "no",
    "fi": "fi",
    "da": "da",
    "cs": "cs",
    "ro": "ro",
    "hu": "hu",
    "el": "el",

    # Southeast Asia
    "th": "th",
    "vi": "vi",
    "id": "id",
    "ms": "ms",
    "tl": "tl",

    # South Asia
    "hi": "hi",
    "ta": "ta",
    "te": "te",
    "ml": "ml",
    "bn": "bn",
    "ur": "ur",

    # Middle East / Cyrillic / Hebrew
    "ar": "ar",
    "fa": "fa",
    "ru": "ru",
    "uk": "uk",
    "he": "he",
    }
    
    if not from_lang or not to_lang:
        return transcribed_text or ""
    if not transcribed_text or not transcribed_text.strip():
        return ""

    src_iso = iso_map.get(from_lang.lower(), "en")
    tgt_iso = iso_map.get(to_lang.lower(), "en")

    if src_iso == tgt_iso:
        return transcribed_text.strip()

    tokenizer.src_lang = src_iso
    encoded = tokenizer(
        transcribed_text,
        return_tensors="pt",
        truncation=True,
        max_length=MAX_INPUT_LENGTH,
    )

    generated_tokens = model.generate(
        **encoded,
        forced_bos_token_id=tokenizer.get_lang_id(tgt_iso),
        max_length=MAX_OUTPUT_LENGTH,
        num_beams=NUM_BEAMS,
    )

    return tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)[0].strip()


if __name__ == "__main__":
    print(f"M2M100 runtime is ready. model_size={MODEL_SIZE}, provider={provider}")
    if len(sys.argv) >= 4:
        print(m2m100(sys.argv[1], sys.argv[2], " ".join(sys.argv[3:])))
