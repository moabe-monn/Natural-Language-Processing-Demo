from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from transformers import TrainingArguments, Trainer
import torch

# 確認: GPUが使えるか
print("GPU available:", torch.cuda.is_available())

# 1. データセットのロード（xlsum 日本語版）
dataset = load_dataset("csebuetnlp/xlsum", "japanese")

# 2. トークナイザーとモデルの読み込み
model_name = "sonoisa/t5-base-japanese"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

# 3. 前処理関数（T5の入力形式に整形）
prefix = "summarize: "

def preprocess_function(examples):
    inputs = [prefix + text for text in examples["text"]]
    model_inputs = tokenizer(inputs, max_length=512, truncation=True, padding="max_length")

    labels = tokenizer(examples["summary"], max_length=128, truncation=True, padding="max_length")
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

# 4. データ前処理
tokenized_datasets = dataset.map(preprocess_function, batched=True, remove_columns=["id", "text", "summary"])

# 5. 学習設定
training_args = TrainingArguments(
    output_dir="./t5-japanese-summary",      # 出力ディレクトリ
    evaluation_strategy="epoch",             # 1エポックごとに評価
    learning_rate=2e-5,
    per_device_train_batch_size=4,           # VRAMに応じて調整
    per_device_eval_batch_size=4,
    num_train_epochs=3,
    weight_decay=0.01,
    save_total_limit=2,
    logging_dir="./logs",
    fp16=torch.cuda.is_available(),          # GPUあれば混合精度ON
    logging_steps=50
)

# 6. Trainerの作成
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    tokenizer=tokenizer,
)

# 7. 学習の開始
trainer.train()

# 8. モデルの保存
trainer.save_model("./t5-japanese-summary-finetuned")
tokenizer.save_pretrained("./t5-japanese-summary-finetuned")

print("ファインチューニング完了 ✅")
