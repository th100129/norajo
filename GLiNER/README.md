1. Background
   GLiNER가 나오기전에는 기존 LLM 모델들은 autoregressive를 사용하여 추론하고 entity를 문장처럼 생성하여 token 생성 속도가 느리며 computing resources를 많이 사용 하였지만 GLiNER은 이를 해결하고자 하여 BERT와 같은 양방향 transformer를 encoder로 사용하였습니다. 이로 인하여 병렬적으로 연산이 가능하여 추론 속도가 빨라졌고 생성기반이 아니라 매칭 기반을 사용하여 연산량을 줄였습니다. 그리고 모든 유형의 entity를 식별할수있는 compact한 모델입니다.

2. Structure
   데이터가 encoder에 들어가면 우선 모든 토큰에 대해 문맥이 반영된 벡터를 출력 하게 됩니다. 그 후 Token representation을 하게 되는 데 여기서는 encoder에서의 출력 값들을 각 토큰 위치에 대응하는 벡터로 변환합니다. 이를 위하여 저는 text와 entity를 분리하여 설계하였고 각각 p벡터와 h벡터를 출력하도록 하였습니다. 매칭을 하기전에 p와 h벡터를 그대로 사용하지 않기 때문에 이들을 각각 q,s 벡터로 만들어주는 FFN을 거치게 됩니다. 다만 span 방식을 사용하기에 FFN을 진행하기전에 h벡터중 시작 위치 벡터와 끝 위치 벡터를 concat하여 주고 FFN을 진행 합니다. 그리고 s와 q는 내적을 통하여 점수를 구하는 매칭을 통하여 점수를 디코더에 보내 최종적으로 엔티티를 선택합니다. 이때 디코더에서는 threshold보다 낮은 점수는 제거하고 남은 후보중 엔티티 후보가 될 span을 선택하고 서로 겹치지 않는 span 만 최종 선택하여 출력합니다.


   
