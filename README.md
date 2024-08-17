# ai-economia

## RAG
### Requirements
datasets


## Fine tunning - Llama3
### Requirements

      pandas
      os
      datasets
      load_dataset
      DatasetDict
      AutoTokenizer
      AutoModelForCausalLM
      TextStreamer
      StoppingCriteria
      StoppingCriteriaList

### Unsloth

unsloth 라이브러리와 관련 디펜던시를 설치하는 과정을 설명합니다.

Colab 환경에서 torch 버전 2.2.1과 호환되지 않는 패키지를 회피하기 위해 unsloth 라이브러리를 별도로 설치합니다.
GPU의 종류(신형 또는 구형)에 따라 조건부로 필요한 패키지들을 설치합니다.
신형 GPU(Ampere, Hopper 등)의 경우, packaging, ninja, einops, flash-attn, xformers, trl, peft, accelerate, bitsandbytes 패키지를 의존성 없이 설치합니다.
구형 GPU(V100, Tesla T4, RTX 20xx 등)의 경우, xformers, trl, peft, accelerate, bitsandbytes 패키지를 의존성 없이 설치합니다.
설치 과정에서 발생하는 출력을 숨기기 위해 %%capture 매직 커맨드를 사용합니다.

### Example

