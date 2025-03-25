# CUDA 12.4 + cuDNN 9 + Ubuntu 22.04
FROM nvidia/cuda:12.4.0-devel-ubuntu22.04

# Ubuntuミラーを日本サーバーに変更
RUN sed -i 's|http://archive.ubuntu.com/ubuntu|http://ftp.jaist.ac.jp/pub/Linux/ubuntu|g' /etc/apt/sources.list

# 必要パッケージを先に入れて tar/xz 解凍対応（wgetも）
RUN apt-get update && apt-get install -y --no-install-recommends \
    wget xz-utils && \
    rm -rf /var/lib/apt/lists/*

# cuDNN アーカイブを取得・展開・コピー
WORKDIR /tmp
RUN wget https://developer.download.nvidia.com/compute/cudnn/redist/cudnn/linux-x86_64/cudnn-linux-x86_64-9.6.0.74_cuda12-archive.tar.xz && \
    tar -xvf cudnn-linux-x86_64-9.6.0.74_cuda12-archive.tar.xz && \
    cp cudnn-linux-x86_64-9.6.0.74_cuda12-archive/lib/* /usr/local/cuda/lib64/ && \
    cp cudnn-linux-x86_64-9.6.0.74_cuda12-archive/include/* /usr/local/cuda/include/ && \
    rm -rf /tmp/*

# LD_LIBRARY_PATH の更新
ENV LD_LIBRARY_PATH="/usr/local/cuda/lib64:$LD_LIBRARY_PATH"

# 必要パッケージをインストール
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip git git-lfs \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# 作業ディレクトリ
WORKDIR /app

# Pythonコマンドへのシンボリックリンク
RUN ln -s /usr/bin/python3 /usr/bin/python

# Pythonライブラリ (PyTorch + Transformers + Optimum + etc.)
# --pre でNightlyビルドをインストール
RUN python -m pip install --upgrade pip && \
    pip install --no-cache-dir \
    faster_whisper==1.1.1\
    Flask==3.1.0\
    gevent==24.11.1\
    optimum[onnxruntime-gpu]==1.24.0\
    torch==2.6.0\
    transformers==4.48.3\
    onnx==1.17.0\
    sentencepiece==0.2.0

# モデルディレクトリ作成 & コピー　#必要に応じてコメントアウト
RUN mkdir -p /app/models
COPY models/onnx-m2m100/ /app/models/onnx-m2m100/
COPY models/models--Systran--faster-whisper-large-v3/ /app/models/models--Systran--faster-whisper-large-v3/
#COPY models/onnx-m2m100-1.2B/ /app/models/onnx-m2m100-1.2B/
#COPY models/models--mobiuslabsgmbh--faster-whisper-large-v3-turbo/ /app/models/models--mobiuslabsgmbh--faster-whisper-large-v3-turbo/

# Pythonコードをコピー
COPY fasterwhisper_m2m100_server.py onnx_m2m100.py /app/

# Server証明書をコピー　#必要に応じて, 別途サーバーの証明書を作成する必要がある。
#COPY server.crt server.key /app/

# ポートを公開
EXPOSE 9000 9443

# サーバー起動
CMD ["python", "fasterwhisper_m2m100_server.py"]
