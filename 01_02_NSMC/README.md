> Template 입니다. 

### AIFFEL Campus Online Code Peer Review Templete
- 코더 : 김정현
- 리뷰어 : 김지원

### SentencePiece를 이용한 네이버 영화 리뷰 Classification
#### 요구사항
1. SentencePiece를 이용하여 모델을 만들기까지의 과정이 정상적으로 진행되었는가?	코퍼스 분석, 전처리, SentencePiece 적용, 토크나이저 구현 및 동작이 빠짐없이 진행되었는가?
2. SentencePiece를 통해 만든 Tokenizer가 자연어처리 모델과 결합하여 동작하는가?	SentencePiece 토크나이저가 적용된 Text Classifier 모델이 정상적으로 수렴하여 80% 이상의 test accuracy가 확인되었다.
3. SentencePiece의 성능을 다각도로 비교분석하였는가?	SentencePiece 토크나이저를 활용했을 때의 성능을 다른 토크나이저 혹은 SentencePiece의 다른 옵션의 경우와 비교하여 분석을 체계적으로 진행하였다.


### PRT(Peer Review Template)
- [ ]  **1. 주어진 문제를 해결하는 완성된 코드가 제출되었나요?**
    - 문제에서 요구하는 최종 결과물이 첨부되었는지 확인
        - **모델 전까지의 분석, 전처리 등에 과정이 빠짐없이 진행되었습니다**
        - **하지만 SentencePiece Tokenizer와 모델과 결합하여 동작하는 지는 확인은 불가능하였다**
    - 문제를 해결하는 완성된 코드란 프로젝트 루브릭 3개 중 2개, 
    퀘스트 문제 요구조건 등을 지칭
        ```
        from tensorflow.keras.preprocessing.text import Tokenizer
        train_ds_mecab = create_dataset_with_mecab(train_set, batch_size=batch_size)
        val_ds_mecab = create_dataset_with_mecab(validation_set, batch_size=batch_size)
        model_mecab = rnn_model(train_ds_mecab, val_ds_mecab)
        ```
        - **본 내용 코드를 보면 mecab와 더불어 여기에 없는 SentencePiece Tokenizer 또한 모델에 대한 인풋 데이터로 완성된 모습을 보여주고 있다**
        - 해당 조건을 만족하는 코드를 캡쳐해 근거로 첨부
    
- [ ]  **2. 전체 코드에서 가장 핵심적이거나 가장 복잡하고 이해하기 어려운 부분에 작성된 
주석 또는 doc string을 보고 해당 코드가 잘 이해되었나요?**
    - 해당 코드 블럭에 doc string/annotation이 달려 있는지 확인
    - 해당 코드가 무슨 기능을 하는지, 왜 그렇게 짜여진건지, 작동 메커니즘이 뭔지 기술.
    - 주석을 보고 코드 이해가 잘 되었는지 확인
        - 잘 작성되었다고 생각되는 부분을 캡쳐해 근거로 첨부합니다.
    - **문장 정규화와 같은 전처리 부분에서 매우 디테일하게 남기었다**    
    - **이 부분외에도 많았다**
```python
    def find_special_characters(dataframe):
    # 한글, 영문, 숫자, 기본 문장부호, 공백을 제외한 패턴 정의
    pattern = re.compile('[^가-힣a-zA-Z0-9.,?! ]')
    
    # 특수문자를 저장할 집합
    special_characters = set()
    
    # 데이터프레임의 'document' 열을 순회하며 특수문자를 찾음
    for document in dataframe['document']:
        # 문자열 타입이 아닌 경우 문자열로 변환
        document = str(document)
        # 정규 표현식을 사용하여 특수문자 찾기
        matches = pattern.findall(document)
        # 찾은 특수문자를 집합에 추가
        special_characters.update(matches)
    
    # 찾은 특수문자 출력
    print(f"특수문자 목록: {special_characters}")
    return special_characters
```  
    
        
- [ ]  **3. 에러가 난 부분을 디버깅하여 문제를 “해결한 기록을 남겼거나” 
”새로운 시도 또는 추가 실험을 수행”해봤나요?**
    - 문제 원인 및 해결 과정을 잘 기록하였는지 확인
    - 문제에서 요구하는 조건에 더해 추가적으로 수행한 나만의 시도, 
    실험이 기록되어 있는지 확인
        - 잘 작성되었다고 생각되는 부분을 캡쳐해 근거로 첨부합니다.
    - **정규화 과정을 전과 후 문장 평균 길이들을 구하여 비교하였다**
```python
# 정규화 적용 후의 문장 평균 길이
normalized_average_length = calculate_average_length(train_data, 'document')
print(f"정규화 전 문장 평균 길이: {original_average_length}")
print(f"정규화 후 문장 평균 길이: {normalized_average_length}")
```
        
- [ ]  **4. 회고를 잘 작성했나요?**
    - 주어진 문제를 해결하는 완성된 코드 내지 프로젝트 결과물에 대해
    배운점과 아쉬운점, 느낀점 등이 기록되어 있는지 확인
    - 전체 코드 실행 플로우를 그래프로 그려서 이해를 돕고 있는지 확인
        - 잘 작성되었다고 생각되는 부분을 캡쳐해 근거로 첨부합니다.
    - **회고는 찾지못하였다**
        
- [ ]  **5. 코드가 간결하고 효율적인가요?**
    - 파이썬 스타일 가이드 (PEP8) 를 준수하였는지 확인
    - 하드코딩을 하지않고 함수화, 모듈화가 가능한 부분은 함수를 만들거나 클래스로 짰는지
    - 코드 중복을 최소화하고 범용적으로 사용할 수 있도록 함수화했는지
        - 잘 작성되었다고 생각되는 부분을 캡쳐해 근거로 첨부합니다.
        
    - **함수화, 모듈가 전처리부터 모델까지 잘 되어있었다. 모델 생섬 함수를 첨부하겠다.**
    ```python
    # 모델 생성 함수
    def rnn_model(train_ds, val_ds):
    # RNN 모델 구축
    model = Sequential()
    model.add(Embedding(input_dim=vocab_size, output_dim=50))  # 임베딩 벡터의 차원은 100으로 설정
    model.add(SimpleRNN(100))  # RNN의 유닛 수는 100으로 설정
    model.add(Dense(1, activation='sigmoid'))  # 출력층

    # 모델 컴파일
    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

    return model
    ```


