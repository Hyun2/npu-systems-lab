# 2단계 -- PyTorch 레퍼런스 (google/gemma-4-E2B-it, bf16)

이후 모든 백엔드가 이 숫자와 비교된다. 원시 데이터는
`bf16-structure.json`에 있고, logits는 `results/logits/v1/`에 있다.

```bash
python -m harness.reference --model google/gemma-4-E2B-it --precision bf16
```

**실행 환경**: RTX 3060 12GB, transformers 5.14.1, torch 2.13.0+cu130,
`Gemma4ForCausalLM` / `Gemma4TextConfig`, eager attention (요청과 실제가
일치), 결정론성 검사 통과, `missing_keys: 0`.

## 텍스트 전용 VRAM -- 계산값 9.3GB, 실측 9.37GB

계획 단계의 9.3GB는 체크포인트 10.25GB에서 vision(~150M)과 audio(~300M)
인코더 파라미터를 뺀 **산술 추정치였고 측정된 적이 없었다.** 실측 결과
추정이 0.8% 안에서 맞았다.

| 항목 | GiB | GB |
|---|---|---|
| 가중치 텐서 합 | 8.621 | 9.26 |
| 로드로 늘어난 디바이스 메모리 | 8.730 | 9.37 |
| 카드 전체 (CUDA 기준) | 11.631 | 12.49 |
| 로드 후 여유 | 2.790 | 3.00 |
| 1849토큰 프롬프트 중 allocator 최대 | 10.440 | 11.21 |

두 가지 관점을 따로 기록한 이유가 여기서 드러난다. allocator 관점
(`memory_allocated`)은 텐서만 세므로 "모델이 얼마인가"에 답하고, 드라이버
관점(`mem_get_info` 차분)은 "카드에서 얼마가 사라지는가"에 답한다. 들어가느냐
마느냐를 판정하는 값은 뒤쪽이다.

**긴 프롬프트가 진짜 제약이다.** 상주 8.62GiB는 여유롭지만, 1849토큰을
한 번에 통과시키면 활성화 텐서가 1.82GiB를 더 쓴다. 여유 2.79GiB 중
약 0.97GiB만 남는다. KV 캐시를 선점하는 백엔드에서는 이 여백이 먼저
사라진다.

## 파라미터 분포 -- 임베딩이 60%다

| 그룹 | 파라미터 | 비중 | 텐서 |
|---|---|---|---|
| per_layer_embeddings (PLE) | 2,390.2M | 51.6% | 108 |
| embeddings | 402.7M | 8.7% | 1 |
| mlp | 1,557.1M | 33.6% | 105 |
| attention | 278.4M | 6.0% | 150 |
| norm | 0.2M | 0.0% | 141 |
| other | 0.0M | 0.0% | 0 |
| **합계** | **4,628.6M** | | |

PLE의 51.6%는 사실상 텐서 하나다.

```
model.embed_tokens_per_layer.weight   [262144, 8960]   2,348.8M   (전체의 50.7%)
model.embed_tokens.weight             [262144, 1536]     402.7M
```

`8960 = 35 layers * 256`. 토큰 하나마다 **레이어별로 256차원**을 따로 들고
있다는 뜻이다. 각 레이어는 `per_layer_projection`(1536 x 256)과
`per_layer_input_gate`(256 x 1536)로 그 조각을 받아 쓴다.

**이것이 "effective 2B"의 정체다.** 저장된 파라미터는 4.63B인데, 그중
2.39B는 토큰마다 **찾아보는(lookup)** 테이블이지 곱해지는 가중치가 아니다.
빼면 2.24B -- 광고된 "2B"와 맞는다. 파라미터 수와 연산량이 갈라지는
지점이 여기다.

**4단계에 대한 예측.** llm-compressor의 공식 Gemma 4 레시피는 `ignore`
목록에 `re:.*embed.*`를 갖고 있다. 그 패턴은 위 두 텐서를 모두 건드리지
않으므로 **파라미터의 60.3%가 bf16으로 남는다.** 나머지 39.7%만 4bit가
된다면 크기는 0.603 + 0.397/4 = 0.70, 즉 30% 감소에 그친다. Google이
배포한 `qat-w4a16-ct`의 실제 감소율은 19%(10.25GB -> 8.35GB)로 같은
방향이다. **이 수치는 추정이며 4단계에서 체크포인트를 직접 열어 확인한다.**

## 어텐션 배치 -- 4 sliding + 1 full, 7회 반복

```
layer  0  1  2  3  4  5  6  7  8  9 ...  34
       s  s  s  s  F  s  s  s  s  F ...   F      (s=sliding 512, F=full)
```

full attention은 레이어 4, 9, 14, 19, 24, 29, 34 -- 정확히 5개마다
하나이고 **마지막 레이어는 full**이다. sliding 28개, full 7개.

부수적으로 확인한 것: `num_key_value_heads=1`(MQA), query head 8개,
`head_dim=256`, `num_kv_shared_layers=20`.

**긴 문맥에서 백엔드 간 차이가 커진다면 여기를 의심한다.** 512토큰
경계는 짧은 프롬프트가 절대 건드리지 않고, 백엔드마다 마스크를 만드는
방식과 window 밖을 자르는 시점이 다르다. 프롬프트 세트에 1849토큰짜리를
넣어둔 이유가 이것이다.

## 프롬프트별 측정

| 프롬프트 | 분류 | 토큰 | latency (s) | allocator 최대 (GiB) |
|---|---|---|---|---|
| short-factual | short | 5 | 0.1488 | 8.634 |
| short-arith | short | 4 | 0.1462 | 8.633 |
| long-technical | long | 1849 | 1.0891 | 10.440 |
| multi-ko | multi | 5 | 0.0466 | 8.634 |
| multi-ja | multi | 3 | 0.0466 | 8.632 |

vocab 262,144 전부에서 fp32로 저장했다. 처음 두 건의 latency가 뒤쪽
짧은 프롬프트의 3배인 것은 워밍업이다 -- 같은 5토큰인 short-factual과
multi-ko를 비교하면 0.149 대 0.047이다. **지연 시간의 baseline으로는
7단계 프로파일링 값을 쓰고 이 표는 쓰지 않는다.**

로드 시간은 10.0초 (페이지 캐시가 더워진 상태).

## 이 단계에서 걸린 함정 두 개

**1. 가중치가 하나도 로드되지 않은 채로 실행이 성공한다.**

체크포인트는 텍스트 타워를 `model.language_model.*`에 저장하는데
`Gemma4ForCausalLM`은 `model.*`을 기대한다. transformers 5.14.1의
`conversion_mapping.py`에는 `gemma4_unified` 항목은 있어도 `gemma4_text`
항목이 없다 (같은 파일의 `qwen3_5_text`에는
`PrefixChange(prefix_to_remove="language_model")`가 등록돼 있다).

그대로 두면 파라미터 24개 그룹이 **무작위 초기화**되고, 모델은
정상적으로 돌아가며 logits를 내놓는다. 그 값은 잡음이다.

```python
key_mapping={r"^model\.language_model\.": "model."}
```

어댑터는 `output_loading_info=True`로 받아 `missing_keys`가 하나라도
있으면 예외를 낸다. `unexpected_keys`는 1400개가 나오는데 이것은 정상이다
-- 두고 온 vision·audio 타워가 그만큼이다.

**2. Triton JIT이 `Python.h` 없이 컴파일에 실패한다.**

RoPE 경로가 `torch._native`를 통해 Triton 커널로 내려가면서 드라이버
초기화 시점에 gcc가 죽는다. gcc도 `libcuda.so.1` 링크도 정상이었고
원인은 `python3.12-dev` 미설치였다. `apt install python3.12-dev`로
해결한다. (`TORCH_DISABLE_NATIVE_JIT=1`로 우회할 수도 있지만, 헤더를
설치하면 이후 단계에서 같은 문제가 재발하지 않는다.)

두 실패 모두 **조용히 잘못된 숫자를 만들 수 있었다는 점**이 공통점이다.
첫 번째는 잡음을 레퍼런스로 저장했을 것이고, 두 번째는 최소한 시끄럽게
죽어줬다.
