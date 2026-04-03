
import os
import random
import numpy as np
import torch

from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    DataCollatorForTokenClassification,
    TrainingArguments,
    Trainer,
)
import evaluate

# =========================
# 1. seed 고정
# =========================
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

# =========================
# 2. label 매핑
# KLUE-NER은 보통 BIO 태그 사용
# =========================
LABELS = [
    "O",
    "B-PS", "I-PS",   # Person
    "B-LC", "I-LC",   # Location
    "B-OG", "I-OG",   # Organization
    "B-DT", "I-DT",   # Date
    "B-TI", "I-TI",   # Time
    "B-QT", "I-QT",   # Quantity
]

label2id = {label: i for i, label in enumerate(LABELS)}
id2label = {i: label for label, i in label2id.items()}

# =========================
# 3. char-level entity span -> BIO token label 변환
# KLUE NER 데이터셋은 sentence, tokens, ner_tags를 확인해서
# 구조가 다를 수 있으므로 실제 컬럼 확인이 중요함.
#
# 여기서는 Hugging Face의 klue/ner 포맷 기준으로
# "tokens", "ner_tags"를 바로 쓰는 baseline 버전으로 작성.
# 만약 실제 확인 결과 컬럼명이 다르면 아래 부분만 맞춰주면 됨.
# =========================
def align_labels_with_tokens(labels, word_ids):
    """
    word_ids: tokenizer(..., is_split_into_words=True) 결과
    labels: 단어 기준 BIO 라벨 id 리스트
    """
    new_labels = []
    previous_word_idx = None

    for word_idx in word_ids:
        if word_idx is None:
            new_labels.append(-100)
        elif word_idx != previous_word_idx:
            new_labels.append(labels[word_idx])
        else:
            # 같은 단어에서 subword가 추가로 나온 경우
            # 일반 baseline에서는 첫 토큰만 학습하고 나머지는 무시
            new_labels.append(-100)
        previous_word_idx = word_idx

    return new_labels

# =========================
# 4. 데이터 로드
# =========================
def load_klue_ner():
    dataset = load_dataset("klue", "ner")
    return dataset

# =========================
# 5. 데이터 구조 확인용 함수
# =========================
def inspect_dataset(dataset):
    print("===== Dataset Structure =====")
    print(dataset)
    print("\n===== Train Example =====")
    print(dataset["train"][0])
    print("\n===== Column Names =====")
    print(dataset["train"].column_names)

# =========================
# 6. 토크나이징 + 라벨 정렬
# =========================
def build_preprocess_function(tokenizer):
    def preprocess_function(examples):
        # KLUE-NER HF 버전은 일반적으로 tokens / ner_tags 형태
        tokenized_inputs = tokenizer(
            examples["tokens"],
            truncation=True,
            is_split_into_words=True,
            max_length=128,
        )

        all_labels = examples["ner_tags"]
        new_labels = []

        for i, labels in enumerate(all_labels):
            word_ids = tokenized_inputs.word_ids(batch_index=i)
            aligned = align_labels_with_tokens(labels, word_ids)
            new_labels.append(aligned)

        tokenized_inputs["labels"] = new_labels
        return tokenized_inputs

    return preprocess_function

# =========================
# 7. metric
# seqeval 사용
# =========================
seqeval = evaluate.load("seqeval")


def compute_metrics(eval_preds):
    logits, labels = eval_preds
    predictions = np.argmax(logits, axis=-1)

    true_predictions = []
    true_labels = []

    for pred, lab in zip(predictions, labels):
        cur_preds = []
        cur_labels = []
        for p, l in zip(pred, lab):
            if l != -100:
                cur_preds.append(id2label[p])
                cur_labels.append(id2label[l])
        true_predictions.append(cur_preds)
        true_labels.append(cur_labels)

    results = seqeval.compute(predictions=true_predictions, references=true_labels)

    return {
        "precision": results["overall_precision"],
        "recall": results["overall_recall"],
        "f1": results["overall_f1"],
        "accuracy": results["overall_accuracy"],
    }

def inspect_dataset(dataset):
    print("===== Dataset Structure =====")
    print(dataset)
    print("\n===== Train Example =====")
    print(dataset["train"][0])
    print("\n===== Column Names =====")
    print(dataset["train"].column_names)

# =========================
# 8. main
# =========================
def main():
    set_seed(42)

    model_name = "klue/bert-base"
    output_dir = "./baseline_klue_bert_ner"

    dataset = load_klue_ner()
    inspect_dataset(dataset)

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # 데이터 컬럼 구조가 예상과 다른 경우 여기서 확인 가능
    # 일반적인 HF klue ner 기준: tokens, ner_tags
    preprocess_function = build_preprocess_function(tokenizer)
    tokenized_dataset = dataset.map(
        preprocess_function,
        batched=True,
        remove_columns=dataset["train"].column_names,
    )

    model = AutoModelForTokenClassification.from_pretrained(
        model_name,
        num_labels=len(LABELS),
        id2label=id2label,
        label2id=label2id,
    )

    data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)

    training_args = TrainingArguments(
        output_dir=output_dir,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="steps",
        logging_steps=50,
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=5,
        weight_decay=0.01,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        report_to="none",
        fp16=torch.cuda.is_available(),
    )

    trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["validation"],
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

    print("\n===== Training Start =====")
    trainer.train()

    print("\n===== Validation Evaluation =====")
    val_result = trainer.evaluate(tokenized_dataset["validation"])
    print(val_result)

    print("\n===== Validation Evaluation =====")
    val_result = trainer.evaluate(tokenized_dataset["validation"])
    print(val_result)

    print("\n===== Save Model =====")
    trainer.save_model(os.path.join(output_dir, "best_model"))
    tokenizer.save_pretrained(os.path.join(output_dir, "best_model"))

    print("\nDone.")


if __name__ == "__main__":
    main()