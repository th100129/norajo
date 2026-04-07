<<<<<<< HEAD
# 연구 계획서
---
## 1. Problem Definition
## - Task
- 본 프로젝트에서는 NER 문제를 다룬다.

---

## - 해결하고자 하는 문제
문장에서 사람(PS), 장소(LC), 시간(DT), 수치(QT) 등의 Entity를 정확하게 추출하는 것이 목표.

특히 한국어 문장에서 다양한 형태로 등장하는 Entity를 정확히 인식하는 것이 주요 과제.

기존 NER 방식은 토큰 단위로 접근하지만, 최근 span 기반 접근이 등장하여 더 정확한 개체 추출이 가능해지고 있다.

본 프로젝트는 다음 질문을 중심으로 문제를 정의한다:

## 단순 BERT 기반 NER과 구조적 확장을 적용한 모델(CRF, span 기반)의 성능 차이는 어떻게 나타나는가?

---
## - 중요성 및 응용

NER은 다음과 같은 다양한 NLP 응용의 핵심 기술이다.
- 정보 검색(Information Retrieval)
- 질의응답 시스템(QA)
- 챗봇 및 대화 시스템
- 문서 자동 요약
- 뉴스/리뷰 분석

특히 한국어는 형태소 변화와 띄어쓰기 문제로 인해 NER 난이도가 높기 때문에 성능 개선의 필요성이 크다.

---

## 2. Background & Baseline
## - 기존 접근 방법

  기존 NER 모델들은 주로 다음과 같은 방식으로 발전해왔다.:

| 모델            | 특징                                  |
| ------------- | ----------------------------------- |
| BiLSTM-CRF    | 순차 기반 전통 모델                         |
| BERT          | Transformer 기반 token classification |
| BERT+CRF      | label dependency 고려                 |
| GlobalPointer | span 기반 병렬 예측                       |
| GLiNER        | generative / span 기반 NER            |



---
## - Baseline Model
본 프로젝트에서는 다음 모델을 baseline으로 설정한다.:

## https://huggingface.co/klue/bert-base 기반 Token Classification

구성:
- BERT encoder
- linear classifier(token-level classification)

선정 이유:
- 가장 기본적인 Transformer 기반 NER 구조
- 추가 구조 없이 순수 BERT 성능 확인 가능
- 이후 구조적 개선 비교 기준으로 적합

---

## 3. Proposed Method

## - 핵심 아이디어

본 연구에서는 동일한 BERT backbone 위에서 구조적 확장을 적용했을 때의 성능 변화를 분석한다.

---

## - 비교 모델 구성

| 모델                 | 설명                        |
| ------------------ | ------------------------- |
| **Baseline: BERT** | 단순 token classification   |
| **BERT + CRF**     | label dependency modeling |
| **GlobalPointer**  | span 기반 엔티티 추출            |
| **GLiNER**         | generative / span 기반 모델   |


---

## - Baseline 대비 차이점

| 항목 | BERT          | BERT+CRF | GlobalPointer | GLiNER     |
| -- | ------------- | -------- | ------------- | ---------- |
| 단위 | token         | token    | span          | span       |
| 구조 | 단순 classifier | CRF 추가   | span matrix   | generative |
| 특징 | 독립 예측         | 순차 의존성   | 병렬 처리         | flexible   |

---

## - Dataset & Preprocessing

- Dateset
  - https://huggingface.co/datasets/klue/klue

---

- 데이터 특성
  - 한국어 뉴스 기반
  - 문자 단위 토큰 제공
  - BIO tagging 방식
  - 다양한 entity 타입 포함(PS, LC, DT, QT 등)
 
---

- 전처리 방법
  1. <entity:label> 형태 annotation 제거
  2. clean sentence 생성
  3. 문자 단위 tokens 및 BIO tag 유지
  4. BERT tokenizer 적용(subword tokenization)
  5. character-level tag -> subword align
  6. special token([CLS], {SEP]) label 제외
  7. json 형태로 저장
   
---

- 기대효과
  각 모델은 다음과 같은 성능 차이를 보일 것으로 예상된다.:
    - BERT -> 기본 성능 기준선
    - BERT + CRF -> label consistency 향상
    - GlobalPointer -> boundary 인식 향상
    - GLiNER -> 유연한 엔티티 표현
     
---

## 4. Experiment Design

## - 수행 실험
  다음 모델을 동일 조건에서 학습 및 비교한다.:
  1. BERT(baseline)
  2. BERT+CRF
  3. GlobalPointer
  4. GLiNER

---

## - 비교 항목

 - 성능 (F1 score)
 -  Precision / Recall
 -  학습 속도
 -  추론 속도
   
---
  
## Evaluation Metrics
- Precision
- Recall
- F1 score

---

## - 검증 방법
- KLUE validation set 사용
- 동일 데이터 split 유지
- 동일 tokenizer 기반 비교
- 모델별 동일 조건 학습

---

## 5. Plan

| 날짜   | 수행 내용               |
| ---- | ------------------- |
| 4/3  | 주제 선정 및 모델, 데이터 서치, 역할분담 |
| 4/6  | 데이터 전처리 및 모델 구조 초안 구성  |
| 4/7  | 모델 완성도 높이기   |
| 4/8  |  전체 모델 비교 실험 및 성능 평가     |
| 4/9  | 결과분석 및 보고서, 발표자료    |

---


    
=======
# GlobalPointer 코드 설명
---

## - 해결하고자 하는 문제

최근 자연어 처리에서 개체명 인식(NER)은 텍스트에서 사람, 장소, 기관과 같은 의미 있는 정보를 추출하는 핵심 기술로 활용되고 있습니다. 기존의 NER 방식은 주로 BIO tagging 기반으로, 각 토큰에 대해 B-PER, I-PER, O와 같은 태그를 예측하는 방식이었습니다. 그러나 이러한 방식은 토큰 단위로 예측을 수행하기 때문에 개체의 시작과 끝을 명확하게 모델링하기 어렵고, 특히 중첩된 개체(Nested Entity)를 처리하는 데 한계가 있습니다.

이러한 문제를 해결하기 위해 등장한 것이 GlobalPointer입니다. GlobalPointer는 기존의 토큰 분류 방식이 아니라, 문장 내의 모든 가능한 span, 즉 시작 위치와 끝 위치의 조합을 직접 예측하는 방식으로 NER 문제를 재정의합니다. 모델의 출력은 (entity_type, start, end) 형태로 구성되며, 이는 특정 엔티티 타입에 대해 해당 span이 엔티티일 확률을 나타냅니다. 이를 통해 모델은 개체를 보다 명확하게 span 단위로 인식할 수 있으며, nested entity도 자연스럽게 처리할 수 있습니다.

구조적으로 보면, 입력 문장은 토크나이징을 거쳐 BERT와 같은 사전학습 언어 모델에 입력됩니다. BERT의 출력은 각 토큰에 대한 contextual embedding이며, 이 벡터들은 GlobalPointer 레이어로 전달됩니다. GlobalPointer에서는 각 토큰에 대해 query와 key 벡터를 생성하고, 이 둘의 내적을 통해 모든 (start, end) 조합에 대한 score를 계산합니다. 이 과정은 torch.einsum을 통해 효율적으로 수행되며, 결과적으로 (batch, entity_type, seq_len, seq_len) 형태의 4차원 텐서가 생성됩니다.

또한, 위치 정보를 효과적으로 반영하기 위해 RoPE(Rotary Position Encoding)를 적용합니다. 이는 단순한 positional embedding이 아니라, query와 key 간의 상대적 위치 관계를 반영할 수 있도록 도와주며, span 간의 관계를 더 잘 학습할 수 있게 합니다. 이후 padding 영역과 end < start인 잘못된 span을 제거하기 위해 mask를 적용하여 유효한 span만 남기게 됩니다.

학습 과정에서는 multilabel categorical crossentropy loss를 사용합니다. 이는 각 span이 특정 엔티티에 해당하는지를 독립적으로 판단하는 multi-label classification 문제로 정의되기 때문입니다. 기존 CRF 기반 모델과 달리, GlobalPointer는 별도의 구조적 제약 없이도 전체 span 관계를 동시에 고려할 수 있다는 장점이 있습니다.

데이터 구성 측면에서도 차이가 있습니다. 기존 BIO 방식에서는 토큰마다 하나의 label을 가지지만, GlobalPointer에서는 (entity_type, seq_len, seq_len) 형태의 label matrix를 사용합니다. 예를 들어, 특정 엔티티가 start=2, end=5라면 해당 위치에 1을 설정하여 span을 표현합니다. 이 방식은 모델이 span 자체를 직접 학습하도록 돕습니다.

추론 단계에서는 모델이 예측한 logits에서 threshold 이상의 값을 가지는 span들을 추출하여 엔티티로 변환합니다. 이때 각 엔티티는 label, start, end, score 형태로 구성되며, score는 해당 span의 confidence를 의미합니다. 하지만 GlobalPointer는 BIO tagging을 직접 출력하지 않기 때문에, 후처리 과정에서 span 정보를 BIO 태그로 변환하는 과정이 필요합니다. 이를 위해 score가 높은 span부터 우선적으로 적용하면서 BIO 형식으로 변환하고, 겹치는 영역은 제거하여 최종 태그를 생성합니다.

최종 출력은 JSON 형태로 구성되며, 원본 텍스트, 토큰 정보, 정답 BIO 태그, 예측 BIO 태그, raw prediction(span 기반), 그리고 각 span의 confidence score까지 포함됩니다. 이를 통해 모델의 예측 결과를 다양한 방식으로 분석하고 활용할 수 있습니다.

정리하자면, GlobalPointer는 기존의 sequence labeling 기반 NER을 span classification 문제로 변환함으로써 보다 유연하고 강력한 표현력을 제공합니다. 특히 CRF 없이도 global dependency를 학습할 수 있으며, nested entity 처리와 병렬 연산 측면에서 큰 장점을 갖습니다. BIO tagging은 모델의 출력이 아니라 단순한 표현 방식이기 때문에, GlobalPointer에서는 이를 후처리 단계에서 변환하여 사용하는 것이 핵심적인 특징입니다.

결론적으로, GlobalPointer는 모든 가능한 span을 동시에 평가하여 개체를 인식하는 구조를 가지며, 이를 통해 기존 방식의 한계를 효과적으로 극복한 NER 모델이라고 할 수 있습니다.

---
>>>>>>> 진주용
