# ai-economia

## RAG
### Requirements
datasets


## Unsloth_Llama3_Fine tunning
### Requirements

      pandas
      os
      json
      torch
      datasets
      load_dataset
      DatasetDict
      AutoTokenizer
      AutoModelForCausalLM
      TextStreamer
      StoppingCriteria
      StoppingCriteriaList
      from unsloth import FastLanguageModel

### Unsloth

`unsloth` 라이브러리와 관련 디펜던시를 설치하는 과정을 설명합니다.

- Colab 환경에서 `torch` 버전 2.2.1과 호환되지 않는 패키지를 회피하기 위해 `unsloth` 라이브러리를 별도로 설치합니다.
- GPU의 종류(신형 또는 구형)에 따라 조건부로 필요한 패키지들을 설치합니다.
        "  - 신형 GPU(Ampere, Hopper 등)의 경우, `packaging`, `ninja`, `einops`, `flash-attn`, `xformers`, `trl`, `peft`, `accelerate`, `bitsandbytes` 패키지를 의존성 없이 설치합니다.
        "  - 구형 GPU(V100, Tesla T4, RTX 20xx 등)의 경우, `xformers`, `trl`, `peft`, `accelerate`, `bitsandbytes` 패키지를 의존성 없이 설치합니다.
- 설치 과정에서 발생하는 출력을 숨기기 위해 `%%capture` 매직 커맨드를 사용합니다.

      %%capture
      # Colab에서 torch 2.2.1을 사용하고 있으므로, 패키지 충돌을 방지하기 위해 별도로 설치해야 합니다.
      !pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
      if major_version >= 8:
          # 새로운 GPU(예: Ampere, Hopper GPUs - RTX 30xx, RTX 40xx, A100, H100, L40)에 사용하세요.
          !pip install --no-deps packaging ninja einops flash-attn xformers trl peft accelerate bitsandbytes
      else:
          # 오래된 GPU(예: V100, Tesla T4, RTX 20xx)에 사용하세요.
          !pip install --no-deps xformers trl peft accelerate bitsandbytes
      pass

### 

### Jsonl 파일 저장

      path = '/content/drive/My Drive/qa.jsonl'
      
      with open('/content/qa.jsonl', 'w', encoding='utf-8') as file:
          for _, row in df.iterrows():
              # JSON 객체 생성
              json_object = {
                  'instruction': row['Question'],  # 'instruction' 키에 질문 저장
                  'input': '',                     # 'input' 키는 현재 비워둠 (필요하다면 수정 가능)
                  'output': row['Answer']          # 'output' 키에 답변 저장
              }
              # JSON 객체를 표준 JSON 문자열로 변환하여 파일에 한 줄씩 작성
              file.write(json.dumps(json_object, ensure_ascii=False) + '\n')
      
        jsonl_file = "/content/qa.jsonl"
      dataset = load_dataset("json", data_files=jsonl_file)

### FastLanguageModel을 사용하여 특정 모듈에 대한 성능 향상 기법을 적용한 모델을 구성

      from unsloth import FastLanguageModel
      import torch
      
      max_seq_length = 4096  # 최대 시퀀스 길이를 설정합니다. 내부적으로 RoPE 스케일링을 자동으로 지원합니다!
      # 자동 감지를 위해 None을 사용합니다. Tesla T4, V100은 Float16, Ampere+는 Bfloat16을 사용하세요.
      dtype = None
      # 메모리 사용량을 줄이기 위해 4bit 양자화를 사용합니다. False일 수도 있습니다.
      load_in_4bit = True
      
      # 4배 빠른 다운로드와 메모리 부족 문제를 방지하기 위해 지원하는 4bit 사전 양자화 모델입니다.

      fourbit_models = [
          "unsloth/mistral-7b-bnb-4bit",
          "unsloth/mistral-7b-instruct-v0.2-bnb-4bit",
          "unsloth/llama-2-7b-bnb-4bit",
          "unsloth/gemma-7b-bnb-4bit",
          "unsloth/gemma-7b-it-bnb-4bit",  # Gemma 7b의 Instruct 버전
          "unsloth/gemma-2b-bnb-4bit",
          "unsloth/gemma-2b-it-bnb-4bit",  # Gemma 2b의 Instruct 버전
          "unsloth/llama-3-8b-bnb-4bit",  # Llama-3 8B
      ]  # 더 많은 모델은 https://huggingface.co/unsloth 에서 확인할 수 있습니다.

      model, tokenizer = FastLanguageModel.from_pretrained(
          # model_name = "unsloth/llama-3-8b-bnb-4bit",
          model_name="beomi/Llama-3-Open-Ko-8B-Instruct-preview",  # 모델 이름을 설정합니다.
          max_seq_length=max_seq_length,  # 최대 시퀀스 길이를 설정합니다.
          dtype=dtype,  # 데이터 타입을 설정합니다.
          load_in_4bit=load_in_4bit,  # 4bit 양자화 로드 여부를 설정합니다.
          # token = "hf_...", # 게이트된 모델을 사용하는 경우 토큰을 사용하세요. 예: meta-llama/Llama-2-7b-hf
      )

### LoRA 어댑터를 추가하여 모든 파라미터 중 단 1% ~ 10%의 파라미터만 업데이트
      model = FastLanguageModel.get_peft_model(
          model,
          r=16,  # 0보다 큰 어떤 숫자도 선택 가능! 8, 16, 32, 64, 128이 권장됩니다.
          lora_alpha=32,  # LoRA 알파 값을 설정합니다.
          lora_dropout=0.05,  # 드롭아웃을 지원합니다.
          target_modules=[
              "q_proj",
              "k_proj",
              "v_proj",
              "o_proj",
              "gate_proj",
              "up_proj",
              "down_proj",
          ],  # 타겟 모듈을 지정합니다.
          bias="none",  # 바이어스를 지원합니다.
          # True 또는 "unsloth"를 사용하여 매우 긴 컨텍스트에 대해 VRAM을 30% 덜 사용하고, 2배 더 큰 배치 크기를 지원합니다.
          use_gradient_checkpointing="unsloth",
          random_state=123,  # 난수 상태를 설정합니다.
          use_rslora=False,  # 순위 안정화 LoRA를 지원합니다.
          loftq_config=None,  # LoftQ를 지원합니다.
      )

### 모델 훈련
from trl import SFTTrainer
from transformers import TrainingArguments

tokenizer.padding_side = "right"  # 토크나이저의 패딩을 오른쪽으로 설정합니다.

### SFTTrainer를 사용하여 모델 학습 설정

      trainer = SFTTrainer(
          model=model,  # 학습할 모델
          tokenizer=tokenizer,  # 토크나이저
          train_dataset=dataset,  # 학습 데이터셋
          eval_dataset=dataset,
          dataset_text_field="text",  # 데이터셋에서 텍스트 필드의 이름
          max_seq_length=max_seq_length,  # 최대 시퀀스 길이
          dataset_num_proc=2,  # 데이터 처리에 사용할 프로세스 수
          packing=False,  # 짧은 시퀀스에 대한 학습 속도를 5배 빠르게 할 수 있음
          args=TrainingArguments(
              per_device_train_batch_size=2,  # 각 디바이스당 훈련 배치 크기
              gradient_accumulation_steps=4,  # 그래디언트 누적 단계
              warmup_steps=5,  # 웜업 스텝 수
              num_train_epochs=3,  # 훈련 에폭 수
              max_steps=100,  # 최대 스텝 수
              do_eval=True,
              evaluation_strategy="steps",
              logging_steps=1,  # logging 스텝 수
              learning_rate=2e-4,  # 학습률
              fp16=not torch.cuda.is_bf16_supported(),  # fp16 사용 여부, bf16이 지원되지 않는 경우에만 사용
              bf16=torch.cuda.is_bf16_supported(),  # bf16 사용 여부, bf16이 지원되는 경우에만 사용
              optim="adamw_8bit",  # 최적화 알고리즘
              weight_decay=0.01,  # 가중치 감소
              lr_scheduler_type="cosine",  # 학습률 스케줄러 유형
              seed=123,  # 랜덤 시드
              output_dir="outputs",  # 출력 디렉토리
          ),
      )

### Example

      ### Instruction:
      0.5인 가구에 대해서 설명해주세요
      
      ### Response:
      0.5인 가구란 1인 가구와 2인 가구의 중간 수준인 0.5인 가구를 말한다. 0.5인 가구는 1인 가구와 2인 가구의 중간 수준으로 1인 가구의 단독 가구와 2인 가구의 부부 가구의 중간 수준을 말한다.<|end_of_text|>
