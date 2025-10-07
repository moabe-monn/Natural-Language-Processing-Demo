from google.cloud import vision
import io
from transformers import pipeline

# === OCRパート ===
client = vision.ImageAnnotatorClient()
path = "/mnt/c/Users/momok/OneDrive/デスクトップ/授業_3y/Project Research/article2.jpg"

with io.open(path, 'rb') as image_file:
    content = image_file.read()

image = vision.Image(content=content)
response = client.text_detection(image=image)
texts = response.text_annotations

if not texts:
    print("テキストが見つかりませんでした。")
    exit()

# OCRで検出された全文（最初の要素が全体）
ocr_text = texts[0].description.strip()
# print("----- OCR結果 -----")
# print(ocr_text)

# === 要約パート ===
text2text_pipeline = pipeline(
    task="summarization",
    model="llm-book/t5-base-long-livedoor-news-corpus",
    device_map="cpu"
)

# プレフィックスを付けてプロンプト作成
prompt = "summarize: " + ocr_text

# 長すぎる場合の対処（モデルは512トークン程度が上限）
MAX_TOKENS = 1024  # 入力が長すぎる場合はカット
if len(prompt) > MAX_TOKENS:
    prompt = prompt[:MAX_TOKENS]

# 要約の実行
res = text2text_pipeline(prompt, max_length=100, min_length=10, do_sample=False)
print("\n----- 要約結果 -----")
print(res[0]["summary_text"])
