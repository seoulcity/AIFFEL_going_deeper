### AIFFEL Campus Online Code Peer Review Templete
- 코더 : 김정현
- 리뷰어 : 임정훈

#### 요구사항

### PRT(Peer Review Template)
- [ ]  **1. 주어진 문제를 해결하는 완성된 코드가 제출되었나요?** 1개 완료
    - 문제에서 요구하는 최종 결과물이 첨부되었는지 확인
    - 문제를 해결하는 완성된 코드란 프로젝트 루브릭 3개 중 2개,
    퀘스트 문제 요구조건 등을 지칭
        - 해당 조건을 만족하는 코드를 캡쳐해 근거로 첨부
        
    1. 모델과 데이터를 정상적으로 불러오고, 작동하는 것을 확인하였다.	klue/bert-base를 NSMC 데이터셋으로 fine-tuning 하여, 모델이 정상적으로 작동하는 것을 확인하였다.
        - 네 사전 훈련된 모델을 사용했고 그것을 모델 학습에 맞게 설정을 조정했습니다.
```python
# 매핑(mapping)을 이용한 다중 처리 기법
hf_dataset = hf_dataset_dict.map(transform, batched=True)
```
    2. Preprocessing을 개선하고, fine-tuning을 통해 모델의 성능을 개선시켰다.	Validation accuracy를 90% 이상으로 개선하였다.
        - 개선을 위해 전처리를 개선하였지만, 90퍼센트는 넘지 못했습니다
```python
# 정규화 대상 문자만 포함하는지 확인하는 함수
def contains_only_allowed_chars(text):
    return re.fullmatch("[가-힣a-zA-Z0-9♥^()!,. ]*", str(text)) is not None

# 조건에 맞는 행만 선택하여 새로운 데이터프레임 생성
filtered_df = df[df['document'].apply(contains_only_allowed_chars)]

# 새로운 데이터프레임의 크기 확인
print(filtered_df.shape)
```
    3. 모델 학습에 Bucketing을 성공적으로 적용하고, 그 결과를 비교분석하였다.	Bucketing task을 수행하여 fine-tuning 시 연산 속도와 모델 성능 간의 trade-off 관계가 발생하는지 여부를 확인하고, 분석한 결과를 제시하였다.
        - buckeing을 성공적으로 적용하였지만, 그에 대한 분석 결과는 없습니다
```python
training_arguments = TrainingArguments(
    output_dir="./results",
    num_train_epochs=3,  # 에포크 수 조정
    per_device_train_batch_size=16,
    per_device_eval_batch_size=64,
    learning_rate=2e-5,  # 학습률 증가
    warmup_steps=500,    # 워밍업 스텝 조정
    weight_decay=0.01,   # 가중치 감쇠 유지
    logging_dir="./logs_dynamic_padding_mod",
    save_steps=10000,
    group_by_length=True,  # 유사한 길이의 데이터를 그룹화
    label_smoothing_factor=0,  # 레이블 스무딩 제거
    logging_strategy="steps",  # 로깅 전략 설정
    logging_steps=500,         # 로깅 스텝 설정
    evaluation_strategy="steps",  # 평가 전략 설정
    eval_steps=500,              # 평가 스텝 설정
)
```


- [ ]  **2. 전체 코드에서 가장 핵심적이거나 가장 복잡하고 이해하기 어려운 부분에 작성된
주석 또는 doc string을 보고 해당 코드가 잘 이해되었나요?** 네
    - 해당 코드 블럭에 doc string/annotation이 달려 있는지 확인
    - 해당 코드가 무슨 기능을 하는지, 왜 그렇게 짜여진건지, 작동 메커니즘이 뭔지 기술.
    - 주석을 보고 코드 이해가 잘 되었는지 확인
        - 잘 작성되었다고 생각되는 부분을 캡쳐해 근거로 첨부합니다.
```python
# 토크나이징 함수 정의 (패딩 제외)
def tokenize_function(examples):
    return tokenizer(examples["document"], truncation=True, padding=False)

# 데이터셋에 토크나이징 함수 적용
tokenized_datasets = hf_dataset_dict.map(tokenize_function, batched=True)

# 데이터셋 형식을 PyTorch 텐서로 설정
tokenized_datasets.set_format("torch", columns=["input_ids", "attention_mask", "label"])

# 데이터셋 분할
hf_train_dataset = tokenized_datasets["train"]
hf_val_dataset = tokenized_datasets["validation"]
hf_test_dataset = tokenized_datasets["test"]

# 데이터 콜레이터 초기화
data_collator = DataCollatorWithPadding(tokenizer=tokenizer, return_tensors="pt")
```

- [ ]  **3. 에러가 난 부분을 디버깅하여 문제를 “해결한 기록을 남겼거나”
”새로운 시도 또는 추가 실험을 수행”해봤나요?** 네
    - 문제 원인 및 해결 과정을 잘 기록하였는지 확인
    - 문제에서 요구하는 조건에 더해 추가적으로 수행한 나만의 시도,
    실험이 기록되어 있는지 확인
        - 잘 작성되었다고 생각되는 부분을 캡쳐해 근거로 첨부합니다.
>num_train_epochs: 모델이 데이터에 충분히 학습되지 않아 underfitting이 발생하는 경우, 학습 에포크 수를 늘리는 것이 도움이 될 수 있습니다. 더 많은 에포크는 모델이 학습 데이터를 더 잘 이해하고, 더 정확한 예측을 할 수 있게 도와줍니다.   
>learning_rate: 학습률은 모델이 얼마나 빠르게 수렴하는지를 결정합니다. 너무 높은 학습률은 수렴을 놓칠 수 있고, 너무 낮은 학습률은 학습 속도를 느리게 할 수 있습니다. 적절한 학습률을 찾는 것은 중요합니다.  
>weight_decay: L2 정규화를 통해 모델의 과적합(overfitting)을 방지할 수 있습니다. 이는 모델이 학습 데이터에 너무 맞춰져 새로운 데이터에 대한 일반화 능력이 떨어지는 것을 방지합니다.  
>warmup_steps: 학습 초기에 학습률을 점진적으로 증가시키는 것은 모델이 안정적으로 수렴하도록 도와줍니다. 이는 특히 큰 학습률을 사용할 때 유용합니다.  
>label_smoothing_factor: 레이블 스무딩은 모델이 너무 확신을 가지고 예측하는 것을 방지하여 과적합을 줄일 수 있습니다. 이는 모델이 더 견고하게 일반화하는 데 도움이 됩니다.  


- [ ]  **4. 회고를 잘 작성했나요?** 아니요
    - 주어진 문제를 해결하는 완성된 코드 내지 프로젝트 결과물에 대해
    배운점과 아쉬운점, 느낀점 등이 기록되어 있는지 확인
    - 전체 코드 실행 플로우를 그래프로 그려서 이해를 돕고 있는지 확인
        - 잘 작성되었다고 생각되는 부분을 캡쳐해 근거로 첨부합니다.
        -보이지 않습니다

- [ ]  **5. 코드가 간결하고 효율적인가요?** 네
    - 파이썬 스타일 가이드 (PEP8) 를 준수하였는지 확인
    - 하드코딩을 하지않고 함수화, 모듈화가 가능한 부분은 함수를 만들거나 클래스로 짰는지
    - 코드 중복을 최소화하고 범용적으로 사용할 수 있도록 함수화했는지
        - 잘 작성되었다고 생각되는 부분을 캡쳐해 근거로 첨부합니다.
        
        
```python
# 변환 함수 정의
def transform_with_max_length(data):
    return tokenizer(
        data['document'],
        truncation=True,
        padding='max_length',
        max_length=60,
        return_token_type_ids=False,
    )

# 매핑(mapping)을 이용한 다중 처리 기법
hf_dataset_max_with_length = hf_dataset_dict.map(transform_with_max_length, batched=True)

# 훈련, 검증, 테스트 데이터셋 분할
hf_train_dataset = hf_dataset_max_with_length['train']
hf_val_dataset = hf_dataset_max_with_length['validation']
hf_test_dataset = hf_dataset_max_with_length['test']
```


추가적인 부분
```python
load_best_model_at_end=True,
```
이 부분을을 추가하면 arguments에 추가하면 가장 좋은 모델을 불러올 수 있습니다.