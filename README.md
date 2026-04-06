# 📊 Baseline Model (BERT Token Classification)
## 🔹 Model
Backbone: klue/bert-base

Task: Named Entity Recognition (NER)

Approach: Token Classification (Softmax)

## 🔹 Dataset
KLUE NER

Train: 21,008 samples

Validation: 5,000 samples

## 🔹 Training Configuration
Epochs: 5

Loss: Cross Entropy

Evaluation Metric: Precision / Recall / F1 / Accuracy (seqeval)

## 🔹 Results (Validation)
## Baseline (KLUE-BERT + Linear Classification)

| Epoch | Train Loss | Valid Loss | Precision | Recall | F1 Score | Accuracy |
|------|-----------|-----------|----------|--------|----------|----------|
| 1 | 0.162367 | 0.167950 | 0.818441 | 0.814659 | 0.816546 | 0.946105 |
| 2 | 0.105366 | 0.151202 | 0.854642 | 0.836966 | 0.845712 | 0.955286 |
| 3 | 0.080413 | 0.147205 | 0.855235 | 0.843706 | 0.849431 | 0.956430 |
| 4 | 0.056857 | 0.155693 | 0.855686 | 0.851609 | 0.853643 | 0.957210 |
| 5 | 0.049319 | 0.164677 | 0.859529 | 0.850271 | 0.854875 | 0.957565 |

## 🔹 Analysis
기본 Token Classification 방식만으로도 F1 ≈ 0.85 수준 확보

Recall 대비 Precision이 약간 높은 경향 → 보수적 예측

Epoch 3 이후 성능 증가폭 감소 → early stopping 가능성 있음

## 🔹 Limitations
Token 단위 예측으로 인해

👉 Entity span 전체를 정확히 잡는 데 한계 존재

BIO tagging 구조 특성상

👉 label dependency를 직접 모델링하지 못함

## 🔹 Role in Project

본 Baseline 모델은 다음 모델들과 비교를 위한 기준으로 사용됨:

BERT + CRF

BERT + GlobalPointer

GLiNER
