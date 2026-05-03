FROM python:3.10-slim-bookworm

WORKDIR /app

ENV PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/app/hf_cache \
    HF_HUB_CACHE=/app/hf_cache/hub \
    LD_LIBRARY_PATH=/usr/local/lib/python3.10/site-packages/nvidia/cublas/lib:/usr/local/lib/python3.10/site-packages/nvidia/cudnn/lib

COPY requirements.txt ./requirements.txt

RUN python -m pip install --upgrade pip setuptools wheel && \
    python -m pip install -r requirements.txt && \
    python - <<'PY'
import onnxruntime as ort
from optimum.onnxruntime import ORTModelForSeq2SeqLM
print("Providers:", ort.get_available_providers())
print("Optimum import OK")
PY

COPY setup_env.json ./setup_env.json
COPY setup_env.py ./setup_env.py
COPY fasterwhisper_m2m100_server.py ./fasterwhisper_m2m100_server.py
COPY onnx_m2m100.py ./onnx_m2m100.py

# If you want to bake existing local models into the image, place a models/
# directory next to this Dockerfile and uncomment the following line.
# setup_env.py will skip downloads/exports when the required files already exist.
# COPY models/ /app/models/

# Create runtime/cache directories before running the setup script.
RUN mkdir -p /app/models /app/hf_cache

# setup_env.json is the single source of truth for model selection, device,
# compute type, ONNX Runtime provider, Hugging Face cache, and server ports.
# This step downloads faster-whisper and exports M2M100 to ONNX only when needed.
RUN python setup_env.py

# Create your own certificate and private key if HTTPS is required, then
# uncomment these lines and rebuild the image.
# COPY server.crt ./server.crt
# COPY server.key ./server.key

EXPOSE 9000 9443

CMD ["python", "fasterwhisper_m2m100_server.py"]
