### AIFFEL Campus Online Code Peer Review Templete
- 코더 : 김정현
- 리뷰어 : 임정훈.


# PRT(Peer Review Template)
- [O]  **1. 주어진 문제를 해결하는 완성된 코드가 제출되었나요?**
    - 문제에서 요구하는 최종 결과물이 첨부되었는지 확인
    - 문제를 해결하는 완성된 코드란 프로젝트 루브릭 3개 중 2개, 
    퀘스트 문제 요구조건 등을 지칭
        - 해당 조건을 만족하는 코드를 캡쳐해 근거로 첨부

> 1. 분류 모델의 accuracy가 기준 이상 높게 나왔는가?	3가지 단어 개수에 대해 8가지 머신러닝 기법을 적용하여 그중 최적의 솔루션을 도출하였다. -> O
```
네 모델별로 결과값들을 모두 출력하였습니다. 그에 대한 솔루션을 도출하였습니다.
```
> 2. 분류 모델의 F1 score가 기준 이상 높게 나왔는가?	Vocabulary size에 따른 각 머신러닝 모델의 성능변화 추이를 살피고, 해당 머신러닝 알고리즘의 특성에 근거해 원인을 분석하였다. -> O
```
네 기준 이상 높게 나왔습니다. 그에 대한 원인에 대해 분석한 부분은 보이지 않습니다.
```
> 3. 딥러닝 모델을 활용해 성능이 비교 및 확인되었는가?	동일한 데이터셋과 전처리 조건으로 딥러닝 모델의 성능과 비교하여 결과에 따른 원인을 분석하였다. -> O
네 확인 되었습니다. 그에 대한 원인 분석으로 자료가 부족해서 학습이 덜 된것 같다고 말하셨습니다.

    
- [0]  **2. 전체 코드에서 가장 핵심적이거나 가장 복잡하고 이해하기 어려운 부분에 작성된 
주석 또는 doc string을 보고 해당 코드가 잘 이해되었나요?**
    - 해당 코드 블럭에 doc string/annotation이 달려 있는지 확인
    - 해당 코드가 무슨 기능을 하는지, 왜 그렇게 짜여진건지, 작동 메커니즘이 뭔지 기술.
    - 주석을 보고 코드 이해가 잘 되었는지 확인
        - 잘 작성되었다고 생각되는 부분을 캡쳐해 근거로 첨부합니다.
> 코드에 주석이 잘 달려 있습니다
```python
# 결과 시각화 함수
def plot_comparison(results):
    labels = list(results.keys())
    test_scores = [results[key]['test_score'] for key in labels]
    times = [results[key]['elapsed_time'] for key in labels]

    x = np.arange(len(labels))

    fig, ax1 = plt.subplots(figsize=(10, 8))

    # 다른 모델들의 정확도 막대 그래프
    bars = ax1.barh(x, test_scores, color='skyblue', label='Test Accuracy')
    # 'bi_lstm_rnn' 모델의 정확도 막대 그래프에 다른 색상 적용
    bars[labels.index('bi_lstm_rnn')].set_color('navy')

    ax1.set_xlabel('Test Accuracy')
    ax1.set_yticks(x)
    ax1.set_yticklabels(labels)
    ax1.invert_yaxis()

    ax2 = ax1.twiny()  # x 축을 공유하는 새 축을 생성
    # 다른 모델들의 시간 라인 그래프
    points = ax2.plot(times, x, 'o', color='salmon', label='Time (s)')
    # 'bi_lstm_rnn' 모델의 시간 라인 그래프에 다른 색상 적용
    ax2.plot(times[labels.index('bi_lstm_rnn')], x[labels.index('bi_lstm_rnn')], 'o', color='darkred', markersize=10, label='bi_lstm_rnn Time (s)')

    ax2.set_xlabel('Time (s)')

    plt.legend()
    plt.show()
```
        
- [O]  **3. 에러가 난 부분을 디버깅하여 문제를 “해결한 기록을 남겼거나” 
”새로운 시도 또는 추가 실험을 수행”해봤나요?**
    - 문제 원인 및 해결 과정을 잘 기록하였는지 확인
    - 문제에서 요구하는 조건에 더해 추가적으로 수행한 나만의 시도, 
    실험이 기록되어 있는지 확인
        - 잘 작성되었다고 생각되는 부분을 캡쳐해 근거로 첨부합니다.
> 모델 학습을 10번 돌렸을 때에는 결과가 아주 좋지 않았다. 10번 시도했을 때에는 스케치의 컬러가 많이 반영되어 노이즈가 심했지만, 높은 epoch를 사용하게 되면 유사한 값이 도출될 것 같다. 이러한 과정을 history라는 값에 loss값들을 저장해서 어떤 과정으로 줄어드는지 확인하면 좋을 것 같다. 이번 프로젝트 자체는 기본적으로 구조가 비슷해서 쉬웠지만, 그에 대한 이론은 아직 체화 하지 못한 것 같다. 최근 과제가 학습 시간이 오래 걸리다 보니 미리미리 해야겠다.
        
- [O]  **4. 회고를 잘 작성했나요?**
    - 주어진 문제를 해결하는 완성된 코드 내지 프로젝트 결과물에 대해
    배운점과 아쉬운점, 느낀점 등이 기록되어 있는지 확인
    - 전체 코드 실행 플로우를 그래프로 그려서 이해를 돕고 있는지 확인
        - 잘 작성되었다고 생각되는 부분을 캡쳐해 근거로 첨부합니다.'
> 초기 코드와 수정작업 대부분을 GPT에 위임해서 진행했다. 방대한 코드 전체를 이해하고 작성하는 것이 애초에 쉽진 않았지만, 방향성을 스스로 설정하고 프롬프트 명령을 작성할 수 있었기 때문에 문제가 생기더라도 빠르게 대응할 수 있었다.  
아마 GPT가 없었다면 내가 할 수 있는 작업범위는 크게 제한되었을 것이다. 아니면 인터넷에서 찾을 수 있는 공개된 코드들을 가져다가 고치는 정도로 진행해야 완성을 시킬 수 있었을 거고.  
이게 내가 공부하는 것이 맞나, 혹은 이렇게 GPT에 의존해서 성장할 수 있을까 의구심이 들다가도, 두달전 처음 아이펠 과정을 시작했던 나와 지금의 모습을 비교하면 모델을 설계하고 방향을 작업 방향을 GPT를 이용해서 빠르게 진행할 수 있다는 것 자체가 대단한 진전이라는 생각이 든다.
        
- [O]  **5. 코드가 간결하고 효율적인가요?**
    - 파이썬 스타일 가이드 (PEP8) 를 준수하였는지 확인
    - 하드코딩을 하지않고 함수화, 모듈화가 가능한 부분은 함수를 만들거나 클래스로 짰는지
    - 코드 중복을 최소화하고 범용적으로 사용할 수 있도록 함수화했는지
        - 잘 작성되었다고 생각되는 부분을 캡쳐해 근거로 첨부합니다.
```python
def run_grid_search(model, params, x_train, y_train, x_test, y_test):
    start_time = time.time()  # 시작 시간 기록
    grid_search = GridSearchCV(model, params, n_jobs=-1, cv=5)
    grid_search.fit(x_train, y_train)
    best_model = grid_search.best_estimator_
    test_score = best_model.score(x_test, y_test)
    end_time = time.time()  # 종료 시간 기록
    elapsed_time = end_time - start_time  # 경과 시간 계산
    return {
        'best_params': grid_search.best_params_,
        'test_score': test_score,
        'elapsed_time': elapsed_time  # 경과 시간 추가
    }
```


# 참고 링크 및 코드 개선
```
# 코드 리뷰 시 참고한 링크가 있다면 링크와 간략한 설명을 첨부합니다.
# 코드 리뷰를 통해 개선한 코드가 있다면 코드와 간략한 설명을 첨부합니다.
```
히트맵같이 시각화 자료를 더 활용했으면 좋았을 것 같습니다.
