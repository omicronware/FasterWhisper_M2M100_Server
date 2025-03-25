#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# PVT Leaf client-server model.
#     omicronware(c): https://www.omicronware.com/
#
import re,os
import torch
from optimum.onnxruntime import ORTModelForSeq2SeqLM
from transformers import M2M100Tokenizer

# ここで ONNX形式モデルを読み込みます。
# 事前に "models/onnx_m2m100" ディレクトリを用意し、その中にエクスポートされたモデルを配置してください。
# 環境変数に応じてモデルを切り替える。
MODEL_SIZE = os.environ.get("M2M_MODEL_SIZE", "418M")  # "418M" or "1.2B"
MODEL_DIR = f"models/onnx-m2m100{'-1.2B' if MODEL_SIZE == '1.2B' else ''}"

try:
    tokenizer = M2M100Tokenizer.from_pretrained(MODEL_DIR,local_files_only=True)
    model = ORTModelForSeq2SeqLM.from_pretrained(MODEL_DIR,local_files_only=True)
except Exception as e:
    raise RuntimeError(f"ONNXモデルの読み込みに失敗しました: {e}")



# 2) 翻訳メイン関数: languages_dict は使わず ISOコード変換テーブルで処理
def m2m100(from_lang, to_lang, transcribed_text):
    """
    from_lang, to_lang : ISO 639-1 コード (例: "ja", "en", "zh-cn" 等)
    transcribed_text   : 翻訳対象テキスト
    """

    # ISOコードから M2M100 が期待する言語コードへの変換テーブル
    # （簡体・繁体の区別は厳密には "zh" が両方扱う）
    iso_map = {
        'en': 'en',
        'ja': 'ja',
        'zh-cn': 'zh',
        'zh': 'zh',
        'zh-tw': 'zh',
        'ko': 'ko',
        'fr': 'fr',
        'de': 'de',
        'es': 'es',
        'it': 'it',
        'pt': 'pt',
        'nl': 'nl',
        'ru': 'ru',
        'ar': 'ar',
        'hi': 'hi',
        #'te': 'te', #M2M100_xxx, Not supported.
        'ml': 'ml',
        'bn': 'bn',
        'ur': 'ur',
        'th': 'th',
        'mn': 'mn',
        'id': 'id',
        'sv': 'sv',
        'vi': 'vi',
        'no': 'no',
        'fi': 'fi',
        'he': 'he',
        'uk': 'uk'
        #必要に応じて追加
    }

    # 見つからなければデフォルトで "en" を使う
    src_iso = iso_map.get(from_lang.lower(), 'en')
    tgt_iso = iso_map.get(to_lang.lower(), 'en')

    # ソース言語を tokenizer にセット
    tokenizer.src_lang = src_iso

    # テキストをトークナイズ
    encoded = tokenizer(transcribed_text, return_tensors="pt")

    # 生成: forced_bos_token_id でターゲット言語を指定
    generated_tokens = model.generate(
        **encoded,
        forced_bos_token_id=tokenizer.get_lang_id(tgt_iso),
        max_length=512,
        num_beams=5,
        no_repeat_ngram_size=2
    )

    # 結果をデコード
    result = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)[0]

    return result.strip()

##############################################################################
# 3) テスト実行例
##############################################################################
if __name__ == "__main__":
    text_jp = "昨日は飲みすぎてしまい、朝起きるのが大変でした。"
    print("翻訳(JA→EN):", m2m100("ja", "en", text_jp))

    text_en = "I drank too much yesterday, so it was hard to wake up in the morning."
    print("翻訳(EN→JA):", m2m100("en", "ja", text_en))
