#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import tempfile
import traceback

from flask import Flask, request, jsonify
from faster_whisper import WhisperModel, BatchedInferencePipeline
from werkzeug.exceptions import HTTPException
from gevent.pywsgi import WSGIServer
import gevent

from setup_env import load_config, get_models_dir, resolve_project_path
import onnx_m2m100

app = Flask(__name__)

CONFIG = load_config()
TRANSCRIBE_CONFIG = CONFIG.get("transcribe", {})
SERVER_CONFIG = CONFIG.get("server", {})

MODEL_NAME = str(TRANSCRIBE_CONFIG.get("FW_MODEL_NAME", "small"))
COMPUTE_TYPE = str(TRANSCRIBE_CONFIG.get("FW_COMPUTE_TYPE", "auto"))
DEVICE = str(TRANSCRIBE_CONFIG.get("FW_DEVICE", "cpu"))
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = get_models_dir(CONFIG)

try:
    print(f"Loading faster-whisper model '{MODEL_NAME}' on {DEVICE} ({COMPUTE_TYPE}) ...")
    model = WhisperModel(
        MODEL_NAME,
        device=DEVICE,
        compute_type=COMPUTE_TYPE,
        download_root=str(MODELS_DIR),
    )
    batched_model = BatchedInferencePipeline(model=model)
    print("Batched model loaded successfully.")
except Exception as e:
    print("Error loading model:", e, file=sys.stderr)
    sys.exit(1)


@app.errorhandler(Exception)
def handle_exception(e):
    if isinstance(e, HTTPException):
        return jsonify({"error": e.name, "details": e.description}), e.code
    tb = traceback.format_exc()
    print(tb, file=sys.stderr)
    return jsonify({"error": "Internal server error.", "details": str(e), "trace": tb}), 500


@app.route("/health", methods=["GET", "HEAD", "OPTIONS"])
def health():
    return jsonify({"status": "ok"}), 200


@app.route("/transcribe", methods=["GET", "POST"])
def transcribe():
    if request.method == "GET":
        return jsonify({"status": "ok"}), 200

    if "audio_file" not in request.files:
        return jsonify({"error": "No audio_file in the request"}), 400

    audio_file = request.files["audio_file"]
    from_language = request.form.get("from_language", None)
    to_language = request.form.get("to_language", None)

    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp:
            audio_file.save(tmp)
            tmp_filename = tmp.name
    except Exception as e:
        return jsonify({"error": "Failed to temporarily save audio_file", "details": str(e)}), 500

    try:
        segments, info = batched_model.transcribe(tmp_filename, language=from_language)
        full_text = "".join(segment.text for segment in segments)
        segments_list = [{"start": s.start, "end": s.end, "text": s.text} for s in segments]
        detected_language = getattr(info, "language", from_language)
        translated_text = ""

        if to_language and detected_language and to_language != detected_language:
            translated_text = onnx_m2m100.m2m100(detected_language, to_language, full_text)

    except Exception as e:
        tb = traceback.format_exc()
        print(tb, file=sys.stderr)
        return jsonify({"error": "Transcription process failed.", "details": str(e), "trace": tb}), 500
    finally:
        try:
            os.remove(tmp_filename)
        except Exception:
            pass

    return jsonify({
        "transcript_text": full_text,
        "translated_text": translated_text,
        "segments": segments_list,
        "language": detected_language,
    }), 200


if __name__ == "__main__":
    host = str(SERVER_CONFIG.get("HOST", "0.0.0.0"))
    http_port = int(SERVER_CONFIG.get("HTTP_PORT", 9000))
    https_port = int(SERVER_CONFIG.get("HTTPS_PORT", 9443))
    cert = resolve_project_path(str(SERVER_CONFIG.get("CERT_FILE", "server.crt")))
    key = resolve_project_path(str(SERVER_CONFIG.get("KEY_FILE", "server.key")))

    http_server = WSGIServer((host, http_port), app)
    https_server = None

    print(f"Starting HTTP server on {host}:{http_port}")
    http_server.start()

    # Create your own certificate and private key if HTTPS is required.
    if cert.exists() and key.exists():
        print(f"Starting HTTPS server on {host}:{https_port}")
        https_server = WSGIServer((host, https_port), app, keyfile=str(key), certfile=str(cert))
        https_server.start()
    else:
        print(f"HTTPS disabled: {cert.name}/{key.name} not found.", file=sys.stderr)

    print("Servers are running. Press Ctrl+C to stop.")
    try:
        gevent.wait()
    except KeyboardInterrupt:
        http_server.stop()
        if https_server is not None:
            https_server.stop()
