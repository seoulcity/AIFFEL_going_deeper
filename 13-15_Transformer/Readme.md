### AIFFEL Campus Online Code Peer Review Templete
- 코더 : 김정현
- 리뷰어 : 서민성

#### 요구사항
- 번역기 모델 학습에 필요한 텍스트 데이터 전처리가 잘 이루어졌다.
    - 데이터 정제, SentencePiece를 활용한 토큰화 및 데이터셋 구축의 과정이 지시대로 진행되었다.
- Transformer 번역기 모델이 정상적으로 구동된다.
    - Transformer 모델의 학습과 추론 과정이 정상적으로 진행되어, 한-영 번역기능이 정상 동작한다.
- 테스트 결과 의미가 통하는 수준의 번역문이 생성되었다.
    - 제시된 문장에 대한 그럴듯한 영어 번역문이 생성되며, 시각화된 Attention Map으로 결과를 뒷받침한다.
     

### PRT(Peer Review Template)
- [ ]  **1. 주어진 문제를 해결하는 완성된 코드가 제출되었나요?**
    - 주어진 문제를 해결하는 완성된 코드가 제출되었습니다.
    - 번역기 모델 학습에 필요한 텍스트 데이터 전처리가 잘 이루어졌다.
        - ``` import sentencepiece as spm
            import os
            
            # Sentencepiece를 활용하여 학습한 tokenizer를 생성합니다.
            def generate_tokenizer(corpus, vocab_size, lang="ko", pad_id=0, bos_id=1, eos_id=2, unk_id=3):
                model_name = f"{lang}_spm"
            
                # 텍스트 파일로 코퍼스 저장
                temp_file = f"{model_name}.temp"
                with open(temp_file, 'w', encoding='utf-8') as f:
                    for line in corpus:
                        f.write(f'{line}\n')
            
                # SentencePiece 모델 학습
                spm.SentencePieceTrainer.Train(
                    f'--input={temp_file} --model_prefix={model_name} '
                    f'--vocab_size={vocab_size} --character_coverage=1.0 '
                    f'--pad_id={pad_id} --bos_id={bos_id} --eos_id={eos_id} --unk_id={unk_id}'
                )
            
                # 생성된 모델 로드
                tokenizer = spm.SentencePieceProcessor()
                tokenizer.Load(f"{model_name}.model")
            
                # 임시 파일 삭제
                os.remove(temp_file)
            
                return tokenizer
            
            
            SRC_VOCAB_SIZE = TGT_VOCAB_SIZE = 20000
            
            eng_corpus = []
            kor_corpus = []
            
            for pair in cleaned_corpus:
                k, e = pair.split("\t")
            
                kor_corpus.append(preprocess_sentence(k))
                eng_corpus.append(preprocess_sentence(e))
            
            ko_tokenizer = generate_tokenizer(kor_corpus, SRC_VOCAB_SIZE, "ko")
            en_tokenizer = generate_tokenizer(eng_corpus, TGT_VOCAB_SIZE, "en")
            en_tokenizer.set_encode_extra_options("bos:eos")
            ```
    - Transformer 번역기 모델이 정상적으로 구동된다.
    - Transformer 모델이 정상적으로 구동되는 것을 확인했습니다.
    - ![스크린샷 2023-11-17 오후 5 39 25](https://github.com/seoulcity/AIFFEL_going_deeper/assets/138687269/2b1abc4a-b1f4-4863-bd7a-e67ef4b3bdfe)
    - 테스트 결과 의미가 통하는 수준의 번역문이 생성되었다.
    - 일부 텍스트에서 번역이 잘 이루어지는 것을 확인했습니다. 추가적인 attention map이 있으면 좋을 것 같습니다. 
        - ``` Input: 일곱 명의 사망자가 발생했다.```
          ```Predicted translation: seven people are dead . ```
        - ![스크린샷 2023-11-17 오후 5 44 38](https://github.com/seoulcity/AIFFEL_going_deeper/assets/138687269/6c2c8a40-5dfa-4d23-b3b3-85befe66d050)


- [ ]  **2. 전체 코드에서 가장 핵심적이거나 가장 복잡하고 이해하기 어려운 부분에 작성된
주석 또는 doc string을 보고 해당 코드가 잘 이해되었나요?**
    - 특정 인덱스에 있는 문장에 대한 토큰화 결과를 시각화하는 코드가 주석처리가 잘 이루어져있었으며 이해하기 좋았습니다.
      ```
      # 원하는 인덱스의 데이터에 대해서만 수행
        indices_to_check = [1, 10000, 30000, 50000]
        
        for index in indices_to_check:
            src_sequence = enc_train[index - 1]  # 인덱스는 1부터 시작하므로 -1
            tgt_sequence = dec_train[index - 1]  # 인덱스는 1부터 시작하므로 -1
        
            # 패딩을 제외하고 원래 문장으로 변환
            src_tokens = [ko_tokenizer.IdToPiece(int(token)) for token in src_sequence if token != 0]
            tgt_tokens = [en_tokenizer.IdToPiece(int(token)) for token in tgt_sequence if token != 0]
        
            print(f"Sample {index}:")
            print("소스 시퀀스 (패딩 포함):", src_sequence)
            print("타겟 시퀀스 (패딩 포함):", tgt_sequence)
            print("소스 문장 (패딩 제외):", " ".join(src_tokens))
            print("타겟 문장 (패딩 제외):", " ".join(tgt_tokens))
            print()
      ```

- [ ]  **3. 에러가 난 부분을 디버깅하여 문제를 “해결한 기록을 남겼거나”
”새로운 시도 또는 추가 실험을 수행”해봤나요?**
    - 텐서의 형태가 맞지 않는 오류에 대한 디버깅을 진행했으며 이에대한 해결을 회고에 기록했습니다.
    - colab의 런타임 유형을 tpu로 변경하여 프로젝트를 진행하려는 시도가 있었으며, 안타깝게도 성공하지 못한 것 같습니다.
    - [TPU 사용법 with colab](https://wikidocs.net/119990)
    

- [ ]  **4. 회고를 잘 작성했나요?**
    - 프로젝트에 대한 전반적인 회고가 잘 작성되어있었으며, 해결해나가는 과정이 잘 작성되어있었습니다.
    ```
    모델 작성을 마치고, 텐서 형태가 맞지 않는다는 에러가 발생해서 GPT의 도움으로 디버깅을 시도했다. 차원 숫자를 모두 출력해서 확인하는 방식으로 코드 수정을 제안받아서 계속 몇 차례 출력문을 집어넣다가, 'enc_in'이 들어가야 할 자리에 'emc_in'이 들어가 있다는 것을 발견했다.
    
    학습을 위해서 코드를 직접 따라치는 과정에서 오타가 생긴 모양이었다.
    
    GPT가 없었다면 아마 발견도 못했을 것 같은데... 학습을 위해서 직접 코딩을 할 때에는 타이핑 후에 무결한 코드를 붙여넣는 방식으로 마무리를 하는 게 좋겠다는 생각이 든다.
    
    그런데 안타깝게도 문제는 오타를 수정해도 해결되지 않았다. 결국 코드를 다시 처음부터 가져와서 고치는 과정을 수행했고, 그러자 문제없이 잘 작동이 되었다.
    
    그리고.. 또다시 오류가 발생했다. 학습 중에 번역결과를 출력하는데 영문이 아니라 계속 한글이 나온다.
    
    거슬러올라가보니 최초에 코퍼스 경로 입력 시에 영문과 한글을 뒤집어놨다. (환장..)
    
    완성된 모델을 사용하고, 그것을 개선하는 방식으로 진행하는 것이 효율적이라는 통찰을 얻었던 프로젝트였다. 
    
    TPU 유닛을 사용하기 위해서 완성된 노트북 파일을 사본으로 만들어, GPT를 이용해 TPU용으로 코드 수정을 여러차례 시도했지만, 잘 되지 않았다. 다음에 오류를 살펴볼 여력이 있을때 차근차근 시도해봐야겠다.
    ```

- [ ]  **5. 코드가 간결하고 효율적인가요?**
    - 파이썬 스타일 가이드 (PEP8) 를 준수하였는지 확인 준수하여 작성하였습니다.
    - 하드코딩을 하지않고 함수화, 모듈화가 가능한 부분은 함수를 만들거나 클래스로 짰는지
    - 코드 중복을 최소화하고 범용적으로 사용할 수 있도록 함수화했는지
        - 잘 작성되었다고 생각되는 부분을 캡쳐해 근거로 첨부합니다.
        - 패딩 결과 확인을 위한 코드를 for문을 통하여 하드코딩을 최소화 하였습니다.
          ```
          # 원하는 인덱스의 데이터에 대해서만 수행
            indices_to_check = [1, 10000, 30000, 50000]
            
            for index in indices_to_check:
                src_sequence = enc_train[index - 1]  # 인덱스는 1부터 시작하므로 -1
                tgt_sequence = dec_train[index - 1]  # 인덱스는 1부터 시작하므로 -1
            
                # 패딩을 제외하고 원래 문장으로 변환
                src_tokens = [ko_tokenizer.IdToPiece(int(token)) for token in src_sequence if token != 0]
                tgt_tokens = [en_tokenizer.IdToPiece(int(token)) for token in tgt_sequence if token != 0]
            
                print(f"Sample {index}:")
                print("소스 시퀀스 (패딩 포함):", src_sequence)
                print("타겟 시퀀스 (패딩 포함):", tgt_sequence)
                print("소스 문장 (패딩 제외):", " ".join(src_tokens))
                print("타겟 문장 (패딩 제외):", " ".join(tgt_tokens))
                print()
          ```
