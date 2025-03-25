#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# PVT Leaf client-server model.
#     omicronware(c): https://www.omicronware.com/
#
import os
import sys
import tempfile
import traceback

from flask import Flask, request, jsonify
from faster_whisper import WhisperModel, BatchedInferencePipeline

# ---- onnx_m2m100.py (M2M100(ONNX)版) をimport
import onnx_m2m100

# gevent を利用した WSGI サーバー
from gevent.pywsgi import WSGIServer
import gevent
import ssl


app = Flask(__name__)

MODEL_NAME   = os.environ.get("FW_MODEL_NAME", "large-v3")  # "large-v3" or "large-v3-turbo"
COMPUTE_TYPE = os.environ.get("FW_COMPUTE_TYPE", "auto")   # "auto" or "int8" or "int8_float16" or "int16" or "float16" or "float32"
DEVICE       = os.environ.get("FW_DEVICE", "cuda")          # "cuda" or "cpu"


try:
    print(f"Loading faster-whisper model '{MODEL_NAME}' on {DEVICE} ({COMPUTE_TYPE}) ...")
    model = WhisperModel(MODEL_NAME, device=DEVICE, compute_type=COMPUTE_TYPE, download_root="models")
    batched_model = BatchedInferencePipeline(model=model)
    print("Batched Model loaded successfully.")
except Exception as e:
    print("Error loading model:", e, file=sys.stderr)
    sys.exit(1)


@app.errorhandler(Exception)
def handle_exception(e):
    """Flask の全ての未処理例外をキャッチする"""
    tb = traceback.format_exc()
    return jsonify({
        "error": "内部サーバーエラーが発生しました。",
        "details": str(e),
        "trace": tb
    }), 500


@app.route('/transcribe', methods=['GET', 'POST'])
def transcribe():
    """
    GET:
      - {"status": "ok"} を返してヘルスチェック

    POST:
      - フォームデータ:
        - audio_file (必須): mp3 等の音声ファイル
        - from_language (任意): Faster-Whisper の認識言語 (未指定で自動判定)
        - to_language (任意): 翻訳先言語 (例: 'en', 'ja')
      - JSON 形式で文字起こし・翻訳結果を返す
    """
    if request.method == 'GET':
        return jsonify({"status": "ok"}), 200

    if 'audio_file' not in request.files:
        return jsonify({"error": "No audio_file in the request"}), 400

    audio_file = request.files['audio_file']
    from_language = request.form.get("from_language", None)
    to_language   = request.form.get("to_language", None)

    # 一時ファイルに音声を保存
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp:
            audio_file.save(tmp)
            tmp_filename = tmp.name
    except Exception as e:
        return jsonify({
            "error": "Failed to temporarily save audio_file",
            "details": str(e)
        }), 500

    try:
        # 文字起こし
        segments, info = batched_model.transcribe(tmp_filename, language=from_language)
        full_text = "".join(segment.text for segment in segments)
        segments_list = [
            {"start": segment.start, "end": segment.end, "text": segment.text}
            for segment in segments
        ]

        detected_language = getattr(info, "language", from_language)

        # 翻訳が必要なら実行
        translated_text = None
        if to_language != detected_language:
            from_lang_code = detected_language
            translated_text = onnx_m2m100.m2m100(from_lang_code, to_language, full_text)
        else:
            translated_text = "\n"

    except Exception as e:
        tb = traceback.format_exc()
        return jsonify({
            "error": "Transcription process failed.",
            "details": str(e),
            "trace": tb
        }), 500
    finally:
        # 一時ファイル削除
        try:
            os.remove(tmp_filename)
        except Exception:
            pass

    # 結果JSON
    result = {
        "transcript_text": full_text,
        "translated_text": translated_text,
        "segments": segments_list,
        "language": detected_language
    }
    return jsonify(result), 200


if __name__ == '__main__':
    SSL_CERT_FILE = "server.crt"
    SSL_KEY_FILE = "server.key"

    # HTTPサーバー (0.0.0.0:9000)
    http_server = WSGIServer(('0.0.0.0', 9000), app)

    # HTTPSサーバー (0.0.0.0:9443)
    try:
        https_server = WSGIServer(
            ('0.0.0.0', 9443),
            app,
            keyfile=SSL_KEY_FILE,
            certfile=SSL_CERT_FILE
        )

        print("Starting HTTP server on port 9000")
        http_server.start()

        print("Starting HTTPS server on port 9443")
        https_server.start()

        print("Servers are running. Press Ctrl+C to stop.")
    except ssl.SSLError as e:
        print(f"SSLエラー: {e}", file=sys.stderr)
    except Exception as e:
        print(f"サーバーの起動時にエラーが発生しました: {e}", file=sys.stderr)
        traceback.print_exc()

    try:
        gevent.wait()
    except KeyboardInterrupt:
        print("\nShutdown requested. Stopping servers...")
        http_server.stop()
        https_server.stop()
        print("Servers stopped successfully.")
        sys.exit(0)
