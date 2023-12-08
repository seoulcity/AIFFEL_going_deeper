### AIFFEL Campus Online Code Peer Review Templete
- 코더 : 김정현
- 리뷰어 : 임정훈

#### 요구사항

1. 기존 데이터셋을 추가 정제하고, generation 성능을 끌어올리기 위한 기법들을 실험해 모델 perfomance를 향상시켜보았는가?	
- [ ] 기존 데이터셋의 문제점을 분석하고 전처리 전략을 수립해 추가 정제를 진행했다. Beam search, Top-k(p) sampling 등 최선의 디코딩 전략을 수립해 향상된 모델 추론 결과를 제시했다. BLEU, ROUGE 등 생성된 텍스트를 평가하기 위한 메트릭을 적용한 정량적인 평가 결과와 주관적인 평가를 비교분석하였다.

2. 새로운 데이터를 수집해 전처리를 수행하여 모델을 재학습시켜보았는가?	
- [ ] 모두의 말뭉치, AI hub 등에 공개된 데이터를 사용해 추가 데이터셋을 구축하기 위한 기준과 근거를 수립했다. ChatGPT API나 다양한 한국어 benchmark 데이터셋을 활용해 Human Feedback 을 대체할 수 있는 아이디어를 구현했다. 위를 바탕으로 SFT, RM, PPO 세 단계에 필요한 각 데이터셋을 적절히 구축하여, 모델 추론 결과와 수립한 가설을 비교해보았다.

3. 학습 전략 또는 foundation model을 변경해 모델을 재학습시켜보았는가?	
- [ ] 더 적절한 Instruction Tuning 기법을 적용해 SFT를 해보거나, Reward Model의 ranking algorithm을 개선해보았다. KoGPT-2가 아닌 다른 모델을 initial model로 사용하여 모델 학습을 성공시켰다. 허깅페이스의 accelerate, bitsandbytes 라이브러리 등을 사용하여 더 큰 스케일의 모델로 ChatGPT를 re-building해 모델 성능을 향상시켰다.


### PRT(Peer Review Template)

- [ ]  **1. 주어진 문제를 해결하는 완성된 코드가 제출되었나요?**
    - 문제에서 요구하는 최종 결과물이 첨부되었는지 확인
    - 문제를 해결하는 완성된 코드란 프로젝트 루브릭 3개 중 2개,
    퀘스트 문제 요구조건 등을 지칭
        - 해당 조건을 만족하는 코드를 캡쳐해 근거로 첨부
    > 1. 기존 데이터셋을 추가 정제하고, generation 성능을 끌어올리기 위한 기법들을 실험해 모델 perfomance를 향상시켜보았는가?	기존 데이터셋의 문제점을 분석하고 전처리 전략을 수립해 추가 정제를 진행했다. Beam search, Top-k(p) sampling 등 최선의 디코딩 전략을 수립해 향상된 모델 추론 결과를 제시했다. BLEU, ROUGE 등 생성된 텍스트를 평가하기 위한 메트릭을 적용한 정량적인 평가 결과와 주관적인 평가를 비교분석하였다.  
    > 비교 분석을 하였지만 성능을 향상시킨 부분은 보이지 않습니다.
    > 프로젝트는 학습 단계에서 완성한 코드의 세부 사항을 조정하는 것이었지만, 시간 부족으로 새로운 내용을 추가하지는 못했다. (세 단계에 걸친 강화학습 기반의 미세조정 과정에서 에포크도 1번씩만 돌려서, 결과적으로 성능이 개선되었는지도 뚜렷하게 확인하기 어려웠다)
    > 2. 새로운 데이터를 수집해 전처리를 수행하여 모델을 재학습시켜보았는가?	모두의 말뭉치, AI hub 등에 공개된 데이터를 사용해 추가 데이터셋을 구축하기 위한 기준과 근거를 수립했다. ChatGPT API나 다양한 한국어 benchmark 데이터셋을 활용해 Human Feedback 을 대체할 수 있는 아이디어를 구현했다. 위를 바탕으로 SFT, RM, PPO 세 단계에 필요한 각 데이터셋을 적절히 구축하여, 모델 추론 결과와 수립한 가설을 비교해보았다.
    > 그런 부분은 보이지 않습니다.
    > 3. 학습 전략 또는 foundation model을 변경해 모델을 재학습시켜보았는가?	더 적절한 Instruction Tuning 기법을 적용해 SFT를 해보거나, Reward Model의 ranking algorithm을 개선해보았다. KoGPT-2가 아닌 다른 모델을 initial model로 사용하여 모델 학습을 성공시켰다. 허깅페이스의 accelerate, bitsandbytes 라이브러리 등을 사용하여 더 큰 스케일의 모델로 ChatGPT를 re-building해 모델 성능을 향상시켰다.
    > 그런 부분은 보이지 않습니다.

- [ ]  **2. 전체 코드에서 가장 핵심적이거나 가장 복잡하고 이해하기 어려운 부분에 작성된
주석 또는 doc string을 보고 해당 코드가 잘 이해되었나요?**
    - 해당 코드 블럭에 doc string/annotation이 달려 있는지 확인
    - 해당 코드가 무슨 기능을 하는지, 왜 그렇게 짜여진건지, 작동 메커니즘이 뭔지 기술.
    - 주석을 보고 코드 이해가 잘 되었는지 확인
        - 잘 작성되었다고 생각되는 부분을 캡쳐해 근거로 첨부합니다.
    > 네 그렇습니다
```python
from chatgpt.trainer.strategies import NaiveStrategy

class GPTRM_custom(RewardModel):

    # 초기화 함수
    def __init__(self,
                 pretrained: Optional[str] = None,
                 config: Optional[GPT2Config] = None,
                 checkpoint: bool = False,
                 lora_rank: int = 0,
                 lora_train_bias: str = 'none',
                 tokenizer=None) -> None:
        # 사전 학습된 모델을 로드하는 경우
        if pretrained is not None:
            model = GPT2Model.from_pretrained(pretrained)
            model.resize_token_embeddings(len(tokenizer))
        # 설정 객체를 사용하여 모델을 초기화하는 경우
        elif config is not None:
            model = GPT2Model(config)
        # 기본 설정으로 모델을 초기화하는 경우
        else:
            model = GPT2Model(GPT2Config())
        # 그래디언트 체크포인팅 활성화
        if checkpoint:
            model.gradient_checkpointing_enable()

        # 값(value) 헤드 추가
        value_head = nn.Linear(model.config.n_embd, 1)
        super().__init__(model, value_head, lora_rank, lora_train_bias)

        # 사전 학습된 모델 정보 저장
        if pretrained is not None:
            self.model = model
            self.pretrained = pretrained

    # 모델 저장 함수
    def save_pretrained(self, dir):
        if self.pretrained is not None:
            self.model.save_pretrained(dir)

# GPT-2 모델과 토크나이저 로드
model = AutoModelForCausalLM.from_pretrained('skt/kogpt2-base-v2')
tokenizer = AutoTokenizer.from_pretrained(
    'skt/kogpt2-base-v2', bos_token='', eos_token='', unk_token='', pad_token='',
    padding_side="right",
    model_max_length=512,
)

# 사용자 정의 모델 초기화 및 CUDA로 이동
with NaiveStrategy().model_init_context():
        model = GPTRM_custom(pretrained='skt/kogpt2-base-v2', lora_rank=0, tokenizer=to
```

- [ ]  **3. 에러가 난 부분을 디버깅하여 문제를 “해결한 기록을 남겼거나”
”새로운 시도 또는 추가 실험을 수행”해봤나요?**
    - 문제 원인 및 해결 과정을 잘 기록하였는지 확인
    - 문제에서 요구하는 조건에 더해 추가적으로 수행한 나만의 시도,
    실험이 기록되어 있는지 확인
        - 잘 작성되었다고 생각되는 부분을 캡쳐해 근거로 첨부합니다.
    > 네 그렇습니다  
    > 학습용 코드를 코랩 환경에서 실행가능하도록 조정하는 데에 시간이 많이 걸렸다. 패키지 의존성이 아이펠 LMS 환경과 달라서 계속 에러가 발생했는데, 덕분에 어디서 의존성 문제가 발생했을 때 라이브러리 버전을 확인하고 그에 맞게 환경 설정을 하는 방법을 배웠다.

- [ ]  **4. 회고를 잘 작성했나요?**
    - 주어진 문제를 해결하는 완성된 코드 내지 프로젝트 결과물에 대해
    배운점과 아쉬운점, 느낀점 등이 기록되어 있는지 확인
    - 전체 코드 실행 플로우를 그래프로 그려서 이해를 돕고 있는지 확인
        - 잘 작성되었다고 생각되는 부분을 캡쳐해 근거로 첨부합니다.
    > 네 잘 작성 되어 있습니다
    > 학습용 코드를 코랩 환경에서 실행가능하도록 조정하는 데에 시간이 많이 걸렸다. 패키지 의존성이 아이펠 LMS 환경과 달라서 계속 에러가 발생했는데, 덕분에 어디서 의존성 문제가 발생했을 때 라이브러리 버전을 확인하고 그에 맞게 환경 설정을 하는 방법을 배웠다.  
    프로젝트는 학습 단계에서 완성한 코드의 세부 사항을 조정하는 것이었지만, 시간 부족으로 새로운 내용을 추가하지는 못했다. (세 단계에 걸친 강화학습 기반의 미세조정 과정에서 에포크도 1번씩만 돌려서, 결과적으로 성능이 개선되었는지도 뚜렷하게 확인하기 어려웠다)  
    다만 기반 모델을 바탕으로, 미세조정을 위한 다운스트림 태크스용 데이터셋을 준비하여 성능 조정 또는 용도 변경을 하는 과정에 대해서 이해할 수 있었고 관련된 주요 개념들에 대한 학습도 좋았다.  
    더 공부가 필요하겠다고 느낀 부분은, 기반 모델을 변경하거나 하이퍼파라미터 등을 조정해서 다른 결과를 만드는 것일 수도 있지만 근본적으로는 이미 기반모델이 머신러닝의 패러다임을 바꾸고 있기 때문에 상용 모델을 좀 더 효과적으로 개선하기 위한 작업 혹은 구체적인 태스크에 맞게 인프라를 구축하는 작업이라고 느낀다.  
    관련된 연구주제를 학습하고 궁극적으로 프로젝트도 이것과 관련된 것을 해보고 싶다.

- [ ]  **5. 코드가 간결하고 효율적인가요?**
    - 파이썬 스타일 가이드 (PEP8) 를 준수하였는지 확인
    - 하드코딩을 하지않고 함수화, 모듈화가 가능한 부분은 함수를 만들거나 클래스로 짰는지
    - 코드 중복을 최소화하고 범용적으로 사용할 수 있도록 함수화했는지
        - 잘 작성되었다고 생각되는 부분을 캡쳐해 근거로 첨부합니다.
    > 네 그렇습니다
    ```python
    def generation(input_text):
    input_ids = tokenizer.encode(input_text, return_tensors='pt').to(
        torch.cuda.current_device())
    outputs = actor.generate(input_ids,
                             max_length=250,
                             do_sample=True,
                             top_k=50,
                             top_p=0.95,
                             num_return_sequences=1)
    output = tokenizer.batch_decode(outputs[0], skip_special_tokens=True)[0]
    print()
    print(output)
    return output

    list_prompt = ['불고기용 고기 한우에요?',
               '리처드 닉슨이 43대 부통령직을 수행한 년도는?',
               '시카고 오헤어 국제공항은 어디에 있어?',
               '오늘 미세먼지 어때?']

    list_prompt = [PROMPT_DICT['prompt_input'].format_map({'prompt' : tmp}) for tmp in list_prompt]

    for input_text in list_prompt:
        output = generation(input_text)
    ```
