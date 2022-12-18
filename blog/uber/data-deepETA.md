# [DeepETA: How Uber Predicts Arrival Times Using Deep Learning](https://www.uber.com/en-KR/blog/deepeta-how-uber-predicts-arrival-times/)

딥러닝을 활용하여 도착 시간을 예측하기
- ETA (Estimated Time Arrival, 예상 도착 시간)

![image-1](https://blog.uber-cdn.com/cdn-cgi/image/width=1250,quality=80,onerror=redirect,format=auto/wp-content/uploads/2022/02/cover_figure.png)

정확한 ETA는 소비자 경험을 좌우함.

ETA의 용도
- 요금 계산
- 픽업 시간 추정
- 라이더 배정
- 배달 플래닝
- ...

전통적 라우팅 방식
- 로드 네트워크를 그래프 내 Weighted edge로 표현되는 작은 로드 세그먼트로 분할함
- 최단 거리 알고리즘을 적용하여 그래프 내 최적 경로를 찾고, 가중치를 더해서 ETA를 계산함

그러나
- 실제 환경과 우리가 이해하는 것 사이에는 간극이 존재한다 (The map is not the territory)
- 로드 그래프는 단지 모델일 뿐, 지형의 상태를 완벽하게 반영하지 못함 (can't perfectly capture)
- 또한 특정 라이더/운전자가 어떤 경로를 선택할지도 알 수 없다

실시간 신호와 함께 기존 데이터를 활용하여, 로드 그래프 예측을 위한 ML 모델 학습으로 보다 정확한 실제 ETA를 예측할 수 있다

---

우버는 지난 수년 동안 ETA 예측을 위해 Gradient-boosted decision tree ensembles 모델을 사용했다. ETA 모델과 학습 데이터는 릴리즈마다 점차 커졌고, Spark 팀은 가장 거대하고 깊은 XGBoost 앙상블 모델을 만드는데 기여함.

결국 데이터 또는 XGBoost 모델 크기를 증가시키는 것이 불가능해졌다 (become untenable)

모델의 크기를 늘리고 정확도 향상을 위해 딥러닝을 활용해보기로 했다
- data-parallel SGD를 이용하여 대규모 데이터셋에 대해서도 쉽게 Scaling이 가능하다
- 딥러닝으로 전환하기 위해 우리는 다음 세 가지 주요 Challenge를 극복해야했다.
    1. Latency - 모델은 늦어도 몇 ms만에 ETA 결과를 반환해야 한다.
    2. Accuracy - MAE (Mean Absolute Error, 평균 절대 오차)는 현재 (incumbent) XGBoost 모델에 비해 엄청난 개선을 보여야 한다.
    3. Generality - 모델은 우버의 모든 비즈니스 라인 (모빌리티 및 배달)에 대하여 ETA 예측을 제공할 수 있어야 한다.

이러한 Challenge를 충족하기 위해 Uber AI는 Uber의 Maps 팀과 함께, 전역적 ETA 예측을 위한 저지연 딥러닝 아키텍처를 개발하는 DeepETA 프로젝트 파트너십을 맺었다.
- [Paper](https://arxiv.org/abs/2206.02127)

## Problem Statement

지난 수년 간 현실의 물리적 모델과 딥러닝을 결합하는 시스템에 대한 관심이 높아졌고, 우버에서는 ETA 예측에 같은 접근을 시도함.

우버의 물리적 모델은 라우팅 엔진
- 맵 데이터와 실시간 교통 측정 정보를 사용
- 두 지점 간 최적 경로를 따르는 세그먼트 단위의 횡단 시간의 합 (a sum of segment-wise traversal times)
- 라우팅 엔진의 ETA와 실제 시간 사이의 오차 (Residual)를 예측하는 ML을 적용
    - hybrid approach ETA post-processing
    - DeepETA도 post-processing 모델의 예시
- 라우팅 엔진을 재구성하는 것보다는 (refactor) post-processing 모델을 업데이트하면서 새로운 데이터를 받아들이고 빠르게 변화하는 비즈니스 요구조건을 수용하는 것이 더 효과적

![image-2](https://blogapi.uber.com/wp-content/uploads/2022/08/figure1-1.png)

ETA 오차를 예측하기 위해 post-processing ML 모델은 여러 특성 (features) 을 고려한다.
- 공간적 (Spatial, e.g. 출발지 및 도착지)
- 시간적 (Temporal, e.g. 요청한 시간 및 실시간 교통 정보) 
- 요청의 종류 (The nature of the request, e.g. Delivery dropoff 또는 Rideshare pickup)

Post-processing 모델은 우버에서 가장 높은 QPS (Queries per second, 초당 쿼리 수)를 갖는 모델이다.
- 빠르면서 ETA 요청에 대해 지연시간을 줄여야 하고
- MAE를 감소시켜 ETA 정확도를 증가시켜야 한다.

## How We Made It Accurate

7가지의 신경망을 학습 및 조절했고, 그 중에서 Self-attention을 활용한 Encoder-Decoder 구조가 가장 좋은 정확성을 나타냈다.
- MLP, NODE, TabNet, Sparsely Gated Mixture-of-Experts, HyperNetworks, Transformer, Linear Transformer
- 여러 Feature Encoding을 실험한 결과, 모델의 모든 Input을 세분화 (Discretize) 하고 임베딩시킨 결과 큰 개선이 있었음

![image-3](https://blog.uber-cdn.com/cdn-cgi/image/width=2160,quality=80,onerror=redirect,format=auto/wp-content/uploads/2022/08/figure2-3.png)

### Encoder w/ Self-Attention

Transformer 구조가 ETA 예측과 같은 Tabular data 문제에도 적용될 수 있을까? Transformer의 혁신은 바로 Self-attention 매커니즘에 있다.
- Self-attention은 Seq2seq Operation으로,
    - Vector Sequence를 Input으로 받고
    - Reweighted Vector Sequence를 만든다

언어 모델에서 각 벡터는 단일 단어 토큰을 나타내지만, DeepETA에서 각 벡터는 단일 Feature (e.g. 출발지 또는 시간)를 나타낸다.

Self-attention는 Tabular dataset에 있는 K개의 feature 간의 pairwise interaction을 발견한다 (uncover)
- Pairwise dot product를 적용한 $K\times K$ attention 행렬
- Feature reweight을 위해 scaled dot-products에 Softmax
- 각 Feature에 Self-attention Layer가 적용될 때, Input 내 나머지 feature를 고려하여 모든 feature에 대한 weighted sum을 출력한다

모든 시간적 및 공간적 feature를 하나의 feature로 굽고 (bake), 중요한 feature들에 초점을 맞춘다. 언어 모델과 다르게, DeepETA에서 feature의 순서는 중요하지 않으므로 positional encoding은 없다.

![image-4](https://blog.uber-cdn.com/cdn-cgi/image/width=2160,quality=80,onerror=redirect,format=auto/wp-content/uploads/2022/08/figure3-4.png)

예를 들어 출발지 A에서 도착지 B로 이동할 때, Self-attention Layer는 주어진 시간, 출발지, 도착지, 교통 상태 등을 고려하여 feature의 중요성을 Scaling한다.

![image-5](https://blog.uber-cdn.com/cdn-cgi/image/width=2160,quality=80,onerror=redirect,format=auto/wp-content/uploads/2022/08/figure4.gif)

## Feature Encoding

### Continuous and Categorical Features

DeepETA 모델은 
- 모든 Categorical feature를 임베딩하고
- Continuous feature의 경우 임베딩 이전에 그룹으로 나눈다 (bucketize).
    - 다소 반직관적이지만 (counterintuitively) 그대로 적용하는 것보다 더 좋은 정확도를 보인다.
    - Bucketizing이 반드시 필요한 것은 아니지만, 학습 과정에서 Input space을 분할하는 과정을 생략할 수 있는 장점이 있을 것이다.
- Gradient Boosted Decision Tree 신경망에서, equal-width bucket보다 quantile bucket을 사용하는 것이 더 좋은 정확도를 보였다.
    - quantile bucket은 entropy를 최대화한다.
        - 고정된 수의 bucket에 대해, 다른 bucketing 방식에 비해 원본 feature value에 대한 가장 중요한 정보를 담는다.

### Geospatial Embeddings

Post-processing models은 출발지와 도착지 정보를 위도와 경도 (latitudes and longitudes)로 받는다.

(...)

![image-6](https://blog.uber-cdn.com/cdn-cgi/image/width=2160,quality=80,onerror=redirect,format=auto/wp-content/uploads/2022/08/figure5-1.png)