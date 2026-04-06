klue_ner_globalpointer_train

NER데이터를 GlobalPointer모델 학습에 맞는 형태로 전처리

전처리 결과는 토큰 단위로 변환, 엔티티 정보를 JSON구조로 저장

GlobalPointer SPAN 예측 방식에 적합

엔티티를 단순한 라벨이 아니라 SPAN정보로 변환

input_ids: 모델 입력

attention_mask: 패딩 마스킹

entities: 정답 span 라벨

원문 텍스트→ 토큰화→ span 엔티티 매핑 → 학습용 데이터셋 생성이 NER전처리의 핵심

klue_ner_globalpointer_valid

한국어 문장을 대상으로 한 개체명 인식 학습용 전처리 데이터

한국어 문장에 대해 토큰화와 개체명 라벨링을 수행한 형태

BERT 계열 모델 또는 NER 모델 학습 및 실험에 활용

CRF

토큰 단위의 라벨 시퀀스를 예측하는 전통적인 순차 라벨링 방식

BIO 태깅 기반 NER에서 많이 사용

비교적 해석이 쉬움, 전통적인 NER 베이스라인으로 자주 사용됨

GLiNER

사전에 고정된 엔티티 라벨셋에 덜 의존하면서, 보다 유연하게 엔티티를 인식할 수 있는 최신 NER 모델

적은 수정으로 다양한 도메인에 적용 가능
