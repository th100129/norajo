# 한국어 NER에서 Token vs Span vs Generative 접근 방식을 비교 분석하는 실험 프로젝트

KLUE NER 기반 다양한 NER 모델 성능 비교 프로젝트
(Token-level vs Span-level vs Generative NER)

## 📌 Overview

본 프로젝트는 한국어 NER(Named Entity Recognition) 태스크에서
서로 다른 구조의 모델들을 비교 분석하는 것을 목표로 합니다.

특히 다음과 같은 접근 방식의 차이를 실험합니다:

Token Classification 기반 (BERT, CRF)

Span-based 방식 (GlobalPointer)

Label-conditioned / Generative NER (GLiNER)


## 🎯 Objectives

KLUE NER 데이터셋을 기반으로 다양한 모델 성능 비교

BIO tagging vs Span-based 접근 방식 비교

모델 구조에 따른 성능 및 특징 분석

추가 데이터셋을 통한 일반화 성능 검증

## 📂 Dataset

## ✅ Main Dataset

KLUE NER dataset

약 26K 문장 규모

Character-level BIO tagging

Entity Types:

DT (Date)

LC (Location)

OG (Organization)

PS (Person)

QT (Quantity)

TI (Time)

## ➕ Additional Dataset (for generalization)

kor-ner-spacy-data

corpus4everyone-klue-korean-NER

## 🤖 Models

### 🔹 Backbone

klue/bert-base (Baseline)

### 🔹 Model Variants

| Model                    | Description                       |
| ------------------------ | --------------------------------- |
| **BERT (Baseline)**      | Token classification 기반           |
| **BERT + CRF**           | BIO 태그 간 의존성 반영                   |
| **BERT + GlobalPointer** | Span-level entity detection       |
| **KRF-BERT**             | 한국어 NER 특화 사전학습 모델                |
| **GLiNER**               | Label-conditioned span prediction |
| **SpaCy (optional)**     | Rule + statistical baseline       |

## 🏗️ Architecture
[KLUE Dataset]
      ↓
[Preprocessing]
      ↓
 ┌───────────────┬───────────────┬───────────────┐
 │ Baseline      │ BERT + CRF    │ GlobalPointer │
 └───────────────┴───────────────┴───────────────┘
          │               │               │
          └────── GLiNER ────────────────┘
      ↓
[Evaluation (F1 Score)]
      ↓
[Comparison & Analysis]
      ↓
[Additional Dataset Experiments]

## ⚙️ Preprocessing

KLUE NER 데이터는 기본적으로 BIO tagging 기반이므로
모델별로 다른 전처리가 필요합니다.

### ✔ Token-based (BERT, CRF)
tokens + ner_tags 그대로 사용
### ✔ Span-based (GlobalPointer)
BIO → (start, end, label) 변환
Token index 기준 span 생성
### ✔ GLiNER
BIO → 문자 기반 span 변환
입력 형태:
{
  "text": "...",
  "labels": ["DT","LC","OG","PS","QT","TI"],
  "entities": [{"start": 0, "end": 3, "label": "PS"}]
}

## 🧩 Modules
| Module             | Description             |
| ------------------ | ----------------------- |
| **Span Generator** | BIO → span 변환           |
| **BERT + CRF**     | Token-level + CRF       |
| **GlobalPointer**  | Span-level 모델           |
| **GLiNER**         | Label-conditioned NER   |
| **Evaluator**      | Precision / Recall / F1 |
| **Comparison**     | 모델 간 성능 비교              |

## 👥 Team
| Role                           | 담당 |
| ------------------------------ | -- |
| Preprocessing (Span Generator) | 김경훈 |
| BERT + CRF                     | 박다현 |
| GlobalPointer                  | 진주용 |
| GLiNER                         | 김인하 |
| Evaluation / Comparison        | 허태희(팀장) |

## 📊 Evaluation

Metric: F1 Score (Entity-level)

## 추가 지표:

Precision

Recall

## 🔍 Key Research Questions

BIO tagging vs Span-based 방식 중 어떤 것이 더 효과적인가?

GlobalPointer는 CRF보다 성능이 좋은가?

GLiNER는 zero-shot/generalization에서 강점을 가지는가?


## 한국어 NER에서 span 방식이 더 유리한가?

## 🚀 Expected Contributions

한국어 NER 모델 구조별 성능 비교

다양한 접근 방식(Token vs Span vs Generative) 분석

실무 적용 관점에서의 모델 선택 가이드 제공

## 📌 Future Work

LLM 기반 NER (GPT-NER, instruction tuning)

Multi-task learning (NER + RE)

Domain-specific NER 확장

## 🛠️ Tech Stack

Python

PyTorch

Hugging Face Transformers / Datasets

sklearn (evaluation)

CUDA / GPU

## ⭐ Notes

KLUE dataset 기반으로 실험 수행

모델별 공정한 비교를 위해 동일한 split 및 평가 기준 사용

