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


    
