# FasterWhisper M2M100 Server

FasterWhisper M2M100 Server は、`faster-whisper` による音声文字起こしと、ONNX Runtime 版 M2M100 による多言語翻訳を組み合わせたローカル実行型の Flask/gevent サーバーです。

外部の音声認識APIや翻訳APIを使わず、ローカル環境または Docker コンテナ内で、音声ファイルの文字起こしと翻訳を実行できます。

---

## Features

- **Local execution**: 音声認識と翻訳をローカルで実行します。
- **Faster Whisper transcription**: `faster-whisper` / CTranslate2 を使用します。
- **M2M100 translation via ONNX Runtime**: M2M100 を ONNX にエクスポートして翻訳します。
- **JSON-based configuration**: モデル、デバイス、ポート、キャッシュ先などを `setup_env.json` で管理します。
- **One-command setup**: `python setup_env.py` で必要なモデルを取得・変換します。
- **Skip existing assets**: 必要ファイルが既にある場合、ダウンロードや ONNX export をスキップします。
- **CPU / CUDA transcription support**: 文字起こしは CPU または CUDA を `setup_env.json` で選択できます。
- **CPU-recommended M2M100**: M2M100 は `CPUExecutionProvider` を標準推奨とします。ただし改良・実験用途のため provider 設定は開放しています。
- **HTTP / optional HTTPS**: HTTP は標準で有効、証明書を配置すれば HTTPS も起動します。

---

## Repository layout

```text
.
├── Dockerfile
├── LICENSE
├── README.md
├── requirements.txt
├── setup_env.json
├── setup_env.json.cuda
├── setup_env.py
├── fasterwhisper_m2m100_server.py
└── onnx_m2m100.py
```

### Main files

| File | Purpose |
|---|---|
| `setup_env.json` | 標準設定ファイルです。CPU実行を基本にしています。 |
| `setup_env.json.cuda` | CUDA文字起こし用の設定例です。M2M100はCPU推奨のままです。 |
| `setup_env.py` | `setup_env.json` に従って faster-whisper モデル取得と M2M100 ONNX export を行います。 |
| `fasterwhisper_m2m100_server.py` | Flask/gevent ベースの HTTP/HTTPS サーバーです。 |
| `onnx_m2m100.py` | ONNX Runtime + M2M100Tokenizer による翻訳モジュールです。 |
| `requirements.txt` | Python依存パッケージです。 |
| `Dockerfile` | Docker実行環境を構築します。 |

---

## Runtime flow

基本フローは次の3ステップです。

```bash
# 1. Edit configuration
# Edit setup_env.json

# 2. Download/export required models
python setup_env.py

# 3. Start server
python fasterwhisper_m2m100_server.py
```

`setup_env.py` は `setup_env.json` を読み込み、以下を実行します。

1. faster-whisper モデルを `models/` に取得します。
2. M2M100 を Hugging Face から取得し、ONNX Runtime 用に `models/onnx-m2m100/` または `models/onnx-m2m100-1.2B/` に export します。
3. 必要ファイルが既に揃っている場合は、対応する処理をスキップします。

---

## Configuration

標準の `setup_env.json` は次の構成です。

```json
{
  "setup": {
    "models_dir": "models",
    "hf_cache_dir": "hf_cache",
    "download_faster_whisper": true,
    "download_m2m100": true
  },
  "transcribe": {
    "FW_MODEL_NAME": "small",
    "FW_COMPUTE_TYPE": "auto",
    "FW_DEVICE": "cpu"
  },
  "translation": {
    "M2M_MODEL_SIZE": "418M",
    "M2M_ORT_PROVIDER": "CPUExecutionProvider",
    "MAX_INPUT_LENGTH": 512,
    "MAX_OUTPUT_LENGTH": 256,
    "NUM_BEAMS": 2
  },
  "server": {
    "HOST": "0.0.0.0",
    "HTTP_PORT": 9000,
    "HTTPS_PORT": 9443,
    "CERT_FILE": "server.crt",
    "KEY_FILE": "server.key"
  }
}
```

### `setup`

| Key | Description |
|---|---|
| `models_dir` | faster-whisper と M2M100 ONNX モデルの保存先です。 |
| `hf_cache_dir` | Hugging Face のキャッシュ先です。 |
| `download_faster_whisper` | `true` の場合、faster-whisper モデルを準備します。 |
| `download_m2m100` | `true` の場合、M2M100 を ONNX export します。 |

### `transcribe`

| Key | Example | Description |
|---|---:|---|
| `FW_MODEL_NAME` | `small`, `large-v3-turbo` | 使用する faster-whisper モデルです。 |
| `FW_COMPUTE_TYPE` | `auto`, `int8`, `float16` | faster-whisper の compute type です。 |
| `FW_DEVICE` | `cpu`, `cuda` | 文字起こしを CPU で行うか CUDA で行うかを指定します。 |

### `translation`

| Key | Example | Description |
|---|---:|---|
| `M2M_MODEL_SIZE` | `418M`, `1.2B` | 使用する M2M100 モデルサイズです。 |
| `M2M_ORT_PROVIDER` | `CPUExecutionProvider` | ONNX Runtime provider です。標準では CPU を推奨します。 |
| `MAX_INPUT_LENGTH` | `512` | tokenizer の最大入力長です。 |
| `MAX_OUTPUT_LENGTH` | `256` | 翻訳生成の最大出力長です。 |
| `NUM_BEAMS` | `2` | beam search の beam 数です。 |

### `server`

| Key | Example | Description |
|---|---:|---|
| `HOST` | `0.0.0.0` | サーバー待受ホストです。 |
| `HTTP_PORT` | `9000` | HTTPポートです。 |
| `HTTPS_PORT` | `9443` | HTTPSポートです。 |
| `CERT_FILE` | `server.crt` | HTTPS用証明書です。存在する場合のみHTTPSを起動します。 |
| `KEY_FILE` | `server.key` | HTTPS用秘密鍵です。存在する場合のみHTTPSを起動します。 |

---

## Python setup

### 1. Create a virtual environment

Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\activate
```

Linux / macOS:

```bash
python -m venv .venv
source .venv/bin/activate
```

### 2. Install dependencies

```bash
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements.txt
```

### 3. Edit `setup_env.json`

CPU transcription example:

```json
"transcribe": {
  "FW_MODEL_NAME": "small",
  "FW_COMPUTE_TYPE": "auto",
  "FW_DEVICE": "cpu"
}
```

CUDA transcription example:

```json
"transcribe": {
  "FW_MODEL_NAME": "large-v3-turbo",
  "FW_COMPUTE_TYPE": "float16",
  "FW_DEVICE": "cuda"
}
```

For CUDA transcription, you can also copy the sample file:

Windows PowerShell:

```powershell
Copy-Item .\setup_env.json.cuda .\setup_env.json -Force
```

Linux / macOS:

```bash
cp setup_env.json.cuda setup_env.json
```

### 4. Prepare models

```bash
python setup_env.py
```

This step may take time on the first run because it downloads the selected models and exports M2M100 to ONNX.

Expected output includes messages such as:

```text
[download] faster-whisper model: small
[export] ONNX M2M100 model: facebook/m2m100_418M
[ready] Environment setup completed.
```

If files already exist, you should see `[skip]` messages.

### 5. Start the server

```bash
python fasterwhisper_m2m100_server.py
```

Default endpoints:

```text
HTTP : http://localhost:9000
HTTPS: https://localhost:9443  # only when server.crt and server.key exist
```

---

## Docker setup

The Dockerfile uses `setup_env.json` as the single source of truth. During image build, it installs Python dependencies and runs:

```bash
python setup_env.py
```

Therefore, edit or select `setup_env.json` before building the image. The selected `setup_env.json` is baked into the image at build time. If you change `setup_env.json` later, rebuild the image or mount/replace the file explicitly.

### Important: Docker GPU option must match `FW_DEVICE`

The Python packages in `requirements.txt` are the same regardless of how you run the server, but Docker does not expose the NVIDIA GPU to a container unless you request it at runtime.

If `setup_env.json` contains:

```json
"FW_DEVICE": "cuda"
```

you must run the container with `--gpus all`. Publishing ports with `-p 9000:9000 -p 9443:9443` only exposes network ports; it does not expose the GPU.

| Configuration | Docker run option | Expected behavior |
|---|---|---|
| `FW_DEVICE: "cpu"` | `docker run -p 9000:9000 -p 9443:9443 ...` | CPU transcription. GPU is not required. |
| `FW_DEVICE: "cuda"` | `docker run --gpus all -p 9000:9000 -p 9443:9443 ...` | CUDA transcription. GPU is required. |
| `FW_DEVICE: "cuda"` without `--gpus all` | Not recommended | CTranslate2/faster-whisper may fail during CUDA initialization. |

M2M100 is separate from this. The default M2M100 setting is `CPUExecutionProvider`, so translation remains CPU-based unless an advanced user changes `M2M_ORT_PROVIDER` and the ONNX Runtime dependencies.

### CPU build and run

Use this path when `FW_DEVICE` is `cpu`.

```bash
docker build -t fasterwhisper-m2m100-server:cpu .
docker run --rm -p 9000:9000 -p 9443:9443 fasterwhisper-m2m100-server:cpu
```

### CUDA transcription build and run

Use the CUDA sample configuration first:

```bash
cp setup_env.json.cuda setup_env.json
```

Then build and run with GPU access:

```bash
docker build -t fasterwhisper-m2m100-server:cuda .
docker run --rm --gpus all -p 9000:9000 -p 9443:9443 fasterwhisper-m2m100-server:cuda
```

On Windows PowerShell:

```powershell
Copy-Item .\setup_env.json.cuda .\setup_env.json -Force
docker build -t fasterwhisper-m2m100-server:cuda .
docker run --rm --gpus all -p 9000:9000 -p 9443:9443 fasterwhisper-m2m100-server:cuda
```

For larger models or high-throughput workloads, these optional Docker flags may improve stability in some environments:

```bash
docker run --rm --gpus all --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 \
  -p 9000:9000 -p 9443:9443 fasterwhisper-m2m100-server:cuda
```

The essential flag for CUDA access is `--gpus all`. The `--ipc` and `--ulimit` options are optional tuning parameters; they do not expose the GPU by themselves.

You can verify Docker GPU access with:

```bash
docker run --rm --gpus all nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi
```

### Baking existing local models into the image

If you already have a local `models/` directory, you can include it in the Docker image.

In `Dockerfile`, uncomment:

```dockerfile
COPY models/ /app/models/
```

Then build:

```bash
docker build -t fasterwhisper-m2m100-server .
```

`setup_env.py` checks required files and skips download/export when the selected model already exists.

### Docker notes

- The Docker image downloads/exports models at build time.
- First build can be slow and image size can become large.
- For frequent development, consider mounting `models/` and `hf_cache/` as volumes, or bake the model once and reuse the image.
- The default M2M100 provider is `CPUExecutionProvider`.
- To experiment with M2M100 GPU execution, you must adjust dependencies and `M2M_ORT_PROVIDER` yourself. The default `requirements.txt` uses CPU `onnxruntime`.
- If the same image succeeds with `--gpus all` but fails without it while `FW_DEVICE` is `cuda`, this is expected. The container without `--gpus all` cannot see the NVIDIA GPU even though the installed Python packages are identical.

---

## API

### GET `/health`

Health check endpoint.

```bash
curl http://localhost:9000/health
```

Response:

```json
{"status":"ok"}
```

### GET `/transcribe`

Compatibility health check endpoint.

```bash
curl http://localhost:9000/transcribe
```

Response:

```json
{"status":"ok"}
```

### POST `/transcribe`

Upload an audio file and receive transcription and optional translation.

Form parameters:

| Name | Required | Description |
|---|---:|---|
| `audio_file` | Yes | Audio file such as mp3, wav, m4a, etc. |
| `from_language` | No | Source language code. If omitted, Whisper detects the language. |
| `to_language` | No | Target language code. If omitted or same as detected language, translation is skipped. |

Example:

```bash
curl -X POST http://localhost:9000/transcribe \
  -F "audio_file=@audio_sample.mp3" \
  -F "from_language=ja" \
  -F "to_language=en"
```

Example response:

```json
{
  "transcript_text": "こんにちは。今日はテストです。",
  "translated_text": "Hello. This is a test today.",
  "segments": [
    {"start": 0.0, "end": 2.5, "text": "こんにちは。今日はテストです。"}
  ],
  "language": "ja"
}
```

---

## HTTPS

HTTPS is optional. Place your own certificate and private key in the project root:

```text
server.crt
server.key
```

Then start the server again:

```bash
python fasterwhisper_m2m100_server.py
```

If both files exist, HTTPS starts on the port configured by `server.HTTPS_PORT`. If they do not exist, the server continues with HTTP only.

---

## Model directories

Default output directories:

```text
models/
├── onnx-m2m100/          # M2M100 418M ONNX export
├── onnx-m2m100-1.2B/     # M2M100 1.2B ONNX export, if selected
└── models--Systran--...  # faster-whisper cache/export directories
```

M2M100 ONNX export is considered complete when files such as the following exist:

```text
encoder_model.onnx
decoder_model.onnx
decoder_with_past_model.onnx
sentencepiece.bpe.model
config.json
tokenizer_config.json
special_tokens_map.json
```

---

## M2M100 provider policy

The default and recommended M2M100 provider is:

```json
"M2M_ORT_PROVIDER": "CPUExecutionProvider"
```

In this project, CPU execution is recommended for M2M100 because it is generally more stable. In some environments, M2M100 ONNX execution on CUDA can cause GPU memory access issues or perform worse than CPU.

However, this repository intentionally does not hard-lock M2M100 to CPU. Advanced users may experiment with other ONNX Runtime providers by editing `setup_env.json` and changing dependencies as needed.

---

## Troubleshooting

### `Run 'python setup_env.py' first.`

M2M100 ONNX files are missing. Run:

```bash
python setup_env.py
```

### `sentencepiece.bpe.model` load error on Windows

If you see an error such as `Illegal byte sequence` while loading `sentencepiece.bpe.model`, avoid OneDrive-managed directories and non-ASCII paths. For example, move the project to:

```text
C:\AI\FasterWhisper_M2M100_Server
```

Then recreate the virtual environment and run setup again.

### CUDA transcription does not start

Check `setup_env.json`:

```json
"FW_DEVICE": "cuda",
"FW_COMPUTE_TYPE": "float16"
```

If you are using Docker, run the container with GPU access:

```bash
docker run --rm --gpus all -p 9000:9000 -p 9443:9443 fasterwhisper-m2m100-server:cuda
```

Also verify Docker GPU access independently:

```bash
docker run --rm --gpus all nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi
```

If GPU access is unavailable, use the CPU configuration:

```json
"FW_DEVICE": "cpu",
"FW_COMPUTE_TYPE": "auto"
```

### `CUDA driver version is insufficient for CUDA runtime version` in Docker

This error can appear when the container tries to start faster-whisper with CUDA but the GPU/driver interface is not exposed to the container. It does not necessarily mean that `requirements.txt` is different.

Common cause:

```bash
docker run -p 9000:9000 -p 9443:9443 fasterwhisper-m2m100-server:cuda
```

with this configuration:

```json
"FW_DEVICE": "cuda"
```

Correct CUDA run command:

```bash
docker run --rm --gpus all -p 9000:9000 -p 9443:9443 fasterwhisper-m2m100-server:cuda
```

Direct Python execution on the host may succeed because the host process can access the NVIDIA driver directly. Docker requires `--gpus all` to pass that access into the container.

The following M2M100 log is not a CUDA transcription error:

```text
ONNX Runtime available providers: ['AzureExecutionProvider', 'CPUExecutionProvider']
Using ONNX Runtime provider: CPUExecutionProvider
```

It only means that M2M100 translation is using CPU ONNX Runtime, which is the recommended default. The CUDA failure in this case is usually from faster-whisper/CTranslate2 because `FW_DEVICE` is `cuda`.

### Docker build takes a long time

The Dockerfile runs `python setup_env.py` at build time. The first build downloads and exports models. This is expected.

To reduce repeated work, either:

- keep Docker layer cache enabled,
- include an existing `models/` directory with `COPY models/ /app/models/`, or
- use mounted volumes in a customized Docker workflow.

---

## Development notes

### Test M2M100 directly

```bash
python onnx_m2m100.py ja en "こんにちは。今日はテストです。"
```

### Test syntax

```bash
python -m py_compile setup_env.py fasterwhisper_m2m100_server.py onnx_m2m100.py
```

### Recommended Git ignore

The following directories should generally not be committed:

```gitignore
.venv/
models/
hf_cache/
__pycache__/
*.pyc
server.crt
server.key
```

---

## License

This repository is released under the MIT License. See `LICENSE` for details.

This project depends on third-party software and models. Users must comply with the licenses and terms of each dependency and model provider, including but not limited to:

- faster-whisper / CTranslate2
- Flask
- gevent
- Hugging Face Transformers
- Hugging Face Optimum
- ONNX Runtime
- SentencePiece
- M2M100 model weights

---

## Disclaimer

This repository is a sample implementation for local transcription and translation. It does not guarantee transcription accuracy, translation accuracy, security, or production readiness. Add authentication, request limits, logging, TLS management, and operational safeguards before using it in production.
