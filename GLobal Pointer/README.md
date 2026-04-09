=======
# GlobalPointer 코드 설명
---

## - 

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
