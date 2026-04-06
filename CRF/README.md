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
