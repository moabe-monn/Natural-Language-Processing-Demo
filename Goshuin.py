from google.cloud import vision
import io
import wikipedia
import google.generativeai as genai
import sys

args = sys.argv

# 言語を日本語に設定
wikipedia.set_lang("ja")

client = vision.ImageAnnotatorClient()

path = "/mnt/c/Users/momok/OneDrive/デスクトップ/授業_3y/Project Research/東京大神宮6.jpg"
with io.open(path, 'rb') as image_file:
    content = image_file.read()

image = vision.Image(content=content)

response = client.document_text_detection(image=image)
texts = response.text_annotations

print('--- 検出されたテキスト ---')

# textsリストが空でないことを確認し、最初の要素（全体テキスト）だけを出力
if texts:
    print(texts[0].description)
else:
    print('テキストが見つかりませんでした。')
    exit()

# エラーがあれば表示
if response.error.message:
    raise Exception(
        '{}\nFor more info on error messages, check: '
        'https://cloud.google.com/apis/design/errors'.format(
            response.error.message))

try:
    # 知りたいトピックのページを取得
    page = wikipedia.page(texts[0].description)

    # ページの全文（.content）を取得
    article_text = page.content
    print("Wikipediaから記事本文を取得しました。")
    # print(article_text) 

except wikipedia.exceptions.PageError:
    print("指定されたページが見つかりませんでした。")
    article_text = None
except wikipedia.exceptions.DisambiguationError as e:
    print("複数の候補が見つかりました。より具体的に指定してください。")
    print(e.options)
    article_text = None

try:
    # 1. 使用するモデルを準備する
    model = genai.GenerativeModel('gemini-2.5-flash')

    # 2. AIに送信するプロンプト（指示）を定義する
    prompt = f"""以下の文章について、歴史の部分と御利益の部分、特徴についての概要を教えてください。

			{article_text}
			"""

    # 3. プロンプトを渡して、コンテンツ（応答）を生成する
    response = model.generate_content(prompt)

    # 4. 生成されたテキストを表示する
    print("Geminiからの回答:")
    print(response.text)

except Exception as e:
    print(f"エラーが発生しました: {e}")