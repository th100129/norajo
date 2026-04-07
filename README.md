# BERT+CRF NER v1

## Overview
BERT+CRF 기반 Named Entity Recognition 모델을 구현하고 학습하였다.

## Model
- Backbone: klue/bert-base
- Structure: BERT → Linear → CRF
- CRF를 통해 token 간 dependency를 고려한 sequence labeling 수행

## Preprocessing
- subword tokenization에 맞춰 label alignment 수행 (word_ids 기반으로 토큰-라벨 정렬)
- [CLS], [SEP], [PAD]는 O로 처리
- padding은 attention_mask로 처리하여 loss 계산에서 제외
- 평가 시 [CLS], [SEP], [PAD] 등 의미 없는 토큰은 제외하고 실제 문장 토큰 기준으로 metric 계산

## Results

| Metric     | Score  |
|------------|--------|
| Accuracy   | 0.9554 |
| Precision  | 0.7624 |
| Recall     | 0.8049 |
| F1-score   | 0.7831 |

## Notes
- 성능 개선 실험 진행 예정

# BERT+CRF NER v2 (Performance Boost)

## Overview

v1에서 발견된 데이터 불균형(O 태그 쏠림) 및 CRF 전이 규칙 학습 부족 문제를 해결하기 위해 손실 함수 가중치 조정과 하이퍼파라미터 튜닝을 적용하여 성능을 개선하였다.

## Improvements & Strategies

Class Weighting (Loss Re-scaling):

학습 데이터 내 O 태그의 압도적 비율로 인한 편향을 제거하기 위해 O 태그의 Loss 가중치는 0.1로 하향, 개체명 태그(B-, I-)는 2.0으로 상향 조정.

모델이 개체명을 놓쳤을 때(False Negative) 더 강한 패널티를 받도록 유도.

CRF Transition Optimization:

CRF 레이어의 Learning Rate를 BERT(2e-5)보다 높은 5e-4로 설정하여 B- → I- 전이 규칙을 더 빠르게 습득하도록 개선.

Differential Learning Rate: Layer별로 최적화된 학습률을 적용하여 사전 학습된 언어 모델의 지식은 보존하면서 상위 레이어의 적응력을 높임.

# Results Comparison

| Metric    | v1 (Baseline) | v2 (Optimized) | Change   |
|-----------|--------------|----------------|----------|
| Accuracy  | 0.9554       | 0.9586         | +0.0032  |
| Precision | 0.7624       | 0.8282         | +0.0658  |
| Recall    | 0.8049       | 0.8233         | +0.0184  |
| F1-score  | 0.7831       | 0.8257         | +0.0426  |


# BERT+CRF NER v2 (Batch Size 16)

## Overview
v2의 최적화 설정을 유지한 상태에서 **batch size를 16으로 설정**하여 추가 실험을 수행하였다.  
그 결과, validation 기준 F1-score가 소폭 상승하여 가장 높은 성능을 기록하였다.

## Additional Setting
- Batch Size: 16
- Epochs: 10
- Best epoch 기준으로 모델 저장

## Training Summary
- Best Valid F1: 0.8276
- Best Valid Accuracy: 0.9582
- Best Valid Precision: 0.8239
- Best Valid Recall: 0.8313

## Results Comparison

| Metric    | v2 (Optimized) | v3 (Batch Size 16) | Change   |
|-----------|--------------:|-------------------:|---------:|
| Accuracy  | 0.9586        | 0.9582             | -0.0004  |
| Precision | 0.8282        | 0.8239             | -0.0043  |
| Recall    | 0.8233        | 0.8313             | +0.0080  |
| F1-score  | 0.8257        | 0.8276             | +0.0019  |
