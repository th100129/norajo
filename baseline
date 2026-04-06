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
| Metric    | Score      |
| --------- | ---------- |
| Precision | 0.8595     |
| Recall    | 0.8503     |
| F1 Score  | **0.8549** |
| Accuracy  | 0.9576     |

## 🔹 Training Trend
| Epoch | F1 Score   |
| ----- | ---------- |
| 1     | 0.8165     |
| 2     | 0.8457     |
| 3     | 0.8494     |
| 4     | 0.8536     |
| 5     | **0.8549** |

👉 Epoch 증가에 따라 안정적으로 성능 향상

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
