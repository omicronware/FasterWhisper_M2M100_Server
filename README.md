# FasterWhisper_M2M100_Server

Faster Whisper + M2M100(ONNX) を組み合わせて、音声ファイルを受け取りリアルタイム文字起こし & 翻訳を行うサーバープログラム

## 概要

このリポジトリは、[faster-whisper](https://github.com/guillaumekln/faster-whisper) を用いて音声の文字起こし（Whisperモデル）を行い、  
[M2M100 (ONNX 版)](https://github.com/facebookresearch/fairseq/tree/main/examples/m2m_100#readme) を使って翻訳をするサーバーアプリケーションを提供します。  
Flask + gevent を利用した Web サーバーとして動作し、HTTP/HTTPS のエンドポイント (`/transcribe`) を通じて文字起こし・翻訳機能を提供します。クライアント側のPVT Leafと完全に互換性があります。

## 特徴
- **完全ローカル動作**: 外部APIを利用せず、プライバシーを確保。
- **GPU対応**: CUDA 12 & cuDNN 9.6.x を推奨。
- **マルチランゲージ対応**: 日本語を含む多言語対応。

---

## 機能一覧

1. **GET** `/transcribe`  
   - ヘルスチェック用。`{"status":"ok"}` を返す。  
2. **POST** `/transcribe`  
   - `audio_file` (音声ファイル: mp3 / wav / m4a 等) をアップロードすると、文字起こし & 翻訳結果を JSON で返す。
   - フォームパラメータ:
     - `from_language`: Whisper で認識させたい言語 (省略可。空の場合は自動判定)
     - `to_language`: 翻訳先言語 (例: `'en'`, `'ja'` など。省略すると翻訳しない)

---

## ファイル構成

- **`fasterwhisper_m2m100_server.py`**  
  Flask と faster-whisper を組み合わせて、WSGI サーバー (`gevent` 使用) で動作するサンプル実装。  
  - `/transcribe` エンドポイントで音声ファイルを受け取り、文字起こしの後、翻訳が必要なら ONNX 版 M2M100 (`onnx_m2m100.py`) を呼び出す。

- **`onnx_m2m100.py`**  
  - ONNX Runtime + M2M100Tokenizer (Transformers) を利用し、多言語翻訳を行うモジュール。  
  - `m2m100(from_lang, to_lang, text)` 関数でテキスト翻訳を行う。

- **`requirements.txt`**  
  - Python パッケージの依存関係リスト。

---

## セットアップ方法

1. **Python 仮想環境の作成 (推奨)**
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # (Windows の場合: .venv\Scripts\activate)
   ```

2. **依存ライブラリのインストール**
   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```
   もしNightly版 PyTorch を使う場合は、`--pre` と専用インデックスを指定するなどの対応が必要です。

3. **M2M100 (ONNX) モデルの配置**  
   `onnx_m2m100.py` の冒頭にある `MODEL_DIR = "models/onnx-m2m100"` に合わせて、事前にエクスポート済みの ONNX M2M100 モデルファイルを `models/onnx-m2m100/` ディレクトリへ配置してください。オリジナルのモデルは、"m2m100_418M"になります。以下にオリジナルからONNXモデルに変換する方法を掲載します。適当な仮想環境上で必要なライブラリをインストールします。
   ```bash
   pip install transformers optimum[onnxruntime] onnx onnxruntime-gpu SentencePiece
   ```
  - 続いて、以下のスクリプトを実行してください。途中ワーニングがでるかもしれません。ダウンロードには十分なRAMが必要です。念のためPowerShellなどで下記の環境変数を設定してください。（例）
   ```bash
   set TEMP=C:\Users\<UserName>\AppData\Local\Temp
   set TMP=C:\Users\<UserName>\AppData\Local\Temp
   ```
  
   ```bash
   from transformers import M2M100ForConditionalGeneration, M2M100Tokenizer
   from optimum.onnxruntime import ORTModelForSeq2SeqLM
   from pathlib import Path

   # モデルIDと保存ディレクトリ
   model_id = "facebook/m2m100_418M"
   save_dir = Path("models/onnx-m2m100")
   save_dir.mkdir(parents=True, exist_ok=True)

   # トークナイザと元モデルを読み込み
   tokenizer = M2M100Tokenizer.from_pretrained(model_id)

   # ONNXモデルに変換・保存
   onnx_model = ORTModelForSeq2SeqLM.from_pretrained(model_id, export=True)
   onnx_model.save_pretrained(save_dir)
   tokenizer.save_pretrained(save_dir)
   ```

4. **Whisper モデル (faster-whisper) の準備**  
   - `fasterwhisper_m2m100_server.py` にある `MODEL_NAME = "large-v3"` を必要に応じて変更。  
   - 事前にモデルファイルをダウンロードして `models/` ディレクトリに配置するか、起動時に自動ダウンロードしても構いません。以下手動でダウンロードする方法を掲載します。適当な仮想環境上で、必要なライブラリをインストールします。
   ```bash
   pip install faster-whisper
   ```
   - 続いて、以下のスクリプトを実行してください。数分後./models/models--Systran--faster-whisper-large-v3/にダウンロードされます。
   ```bash
   from faster_whisper import WhisperModel
   import os

   # モデル名と保存先パス
   model_name = "large-v3"
   save_dir = "./models"

   # モデル保存先を設定
   os.makedirs(save_dir, exist_ok=True)

   print(f"Downloading '{model_name}' model into '{save_dir}'...")

   # モデルを初回だけダウンロード（2回目以降はローカルキャッシュから）
   model = WhisperModel(model_name, download_root=save_dir)

   print("ダウンロード完了！")
   ```

  - なお、このサーバーはいくつかの環境変数を設定することで、M2M100のモデルやfaster-whisperのモデルを変更することができます。詳しくはソースコードを見てください。
  - モデルを高精度のM2M100_1.2Bに変更したい場合は、上記スクリプトを適切に変更（418M → 1.2B, models/onnx-m2m100 → models/onnx-m2m100-1.2Bに変更）
  - PowerShellなどで環境変数を設定する。
   ```bash
   $env:M2M_MODEL_SIZE="1.2B"  # PowerShell の場合
   M2M_MODEL_SIZE=1.2B python fasterwhisper_m2m100_server.py (Linux/MacOS)
   ```

5. **サーバー起動**
   ```bash
   python fasterwhisper_m2m100_server.py
   ```
   - HTTP サーバーが `0.0.0.0:9000` で待機し、HTTPS サーバーが `0.0.0.0:9443` で待機 (証明書があれば) します。

---

## 使い方

- **ヘルスチェック**  
   ```bash
   curl http://localhost:9000/transcribe
   ```
   → `{"status":"ok"}` が返れば正常。

- **音声アップロード (POST)**  
   ```bash
   curl -X POST http://localhost:9000/transcribe \
        -F "audio_file=@audio_sample.mp3" \
        -F "from_language=ja" \
        -F "to_language=en"
   ```
   → JSON 形式で文字起こし (`transcript_text`) と翻訳 (`translated_text`) を返す。

---

## ライセンス

このリポジトリ内のソースコードは [MIT License](#MIT-LICENSE) のもとで公開されています。  
ただし、本リポジトリが使用しているライブラリはそれぞれ独自のライセンスに従います。

### 主要ライブラリのライセンス一覧

- **[faster-whisper](https://github.com/guillaumekln/faster-whisper)**  
  [Apache License 2.0](https://github.com/guillaumekln/faster-whisper/blob/main/LICENSE)
- **[Flask](https://palletsprojects.com/p/flask/)**  
  [BSD-3-Clause](https://github.com/pallets/flask/blob/main/LICENSE.rst)
- **[gevent](https://www.gevent.org/)**  
  [MIT License](https://github.com/gevent/gevent/blob/master/LICENSE)
- **[optimum](https://github.com/huggingface/optimum)**  
  [Apache License 2.0](https://github.com/huggingface/optimum/blob/main/LICENSE)
- **[PyTorch](https://github.com/pytorch/pytorch)**  
  [BSD-3-Clause](https://github.com/pytorch/pytorch/blob/master/LICENSE)
- **[Transformers](https://github.com/huggingface/transformers)**  
  [Apache License 2.0](https://github.com/huggingface/transformers/blob/main/LICENSE)
- **[ONNX](https://github.com/onnx/onnx)**  
  [Apache License 2.0](https://github.com/onnx/onnx/blob/main/LICENSE)
- **[SentencePiece](https://github.com/google/sentencepiece)**  
  [Apache License 2.0](https://github.com/google/sentencepiece/blob/master/LICENSE)

その他のライブラリについては `requirements.txt` などをご参照ください。  
本リポジトリの利用者は、それぞれのライブラリが示すライセンス条項にも従う必要があります。

---
## 環境要件
- **OS**: Windows 10 Pro 22H2 以上（Linux も可）
- **CUDA**: 12 以上（推奨）
- **cuDNN**: 9.6.x 以上（推奨）
- **Python**: 3.9 以上
- **GPU**: NVIDIA GPU（推奨）

## 制限事項

- 本プロジェクトは試験的なサンプル実装です。本番環境での利用は自己責任で行ってください。  
- セキュリティ・認証などは最小限の設定となっています。実運用時は HTTPS 証明書や API 認証を適切に行う必要があります。  
- Whisper および M2M100 の翻訳精度は完璧ではありません。内容の正確さを保証するものではありません。

---

## 開発者向け

### デバッグ方法

- `python fasterwhisper_m2m100_server.py` でサーバーを起動後に、ブラウザや `curl` で `/transcribe` をテストします。  
- 例外が発生した場合、Flask のエラーハンドラが JSON 形式で返し、stderr にトレースバックが表示されます。

### Dockerイメージのビルド
- とりあえず用意しました。
- 別途 Dockerfileを環境に合わせて修正してください。上記の各モデルファイル(M2M100_418M, faster-whisper/large-v3)を適切なディレクトリにコピーしてからビルドしてください。

