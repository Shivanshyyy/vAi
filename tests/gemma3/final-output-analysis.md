# Gemma3 demo.cpp / demo.py 최종 출력 분석 — 2026-09-28

## 결론

현재 출력 차이의 핵심 구현 오류는 **비전 인코더에 causal attention이 적용되는 것**이다.
샘플링 정책도 서로 다르고, Python 출력 코드에는 첫 생성 토큰 누락 및 종료 이후 생성 문제가 있다.
전처리만 일치시켜서는 이 차이가 없어지지 않는다.

이번 요청에서는 원인 분석을 위해 임시 실행 파일만 변경했다. 모델 및 데모 소스에 아래 수정안을 적용하지 않았다.

## 실행 환경과 통제 조건

- RTX 5080: C++ Vulkan 및 Python BF16 CUDA. Vulkan은 `VK_DRIVER_FILES=/usr/share/vulkan/icd.d/nvidia_icd.json`으로 지정했다.
- PyTorch 2.14.0+cu130, Torchvision 0.29.0+cu130, Transformers 5.17.0.
- 같은 로컬 Gemma3 4B 체크포인트, `data/gemma3/image.jpg`, 질문 `Where is he looking?`.
- C++ 기본: FP16 가중치/KV, FP32 I/O, top-k 40, temperature 1.
- Python 기본: BF16, argmax, 64 decode steps.
- 레이어 비교에서는 Python의 `pixel_values.npy`를 공유하고, prompt 273개 ID가 전부 같음을 검증했다.
- C++ decode에는 Python BF16의 `steps.ids.npy` 64개를 그대로 넣었다. 토큰 이력 차이를 제거한 teacher-forcing 비교다.
- 별도 Python FP32 기준은 CPU에서 동일 BF16 토큰 이력으로 16 decode steps를 실행했다. GPU BF16과 CPU FP32 비교는 dtype뿐 아니라 backend 차이도 포함한다.

## 실제 최종 출력

기본 C++:

> It's tricky to say for sure, but based on the image, it appears the man is looking towards the **distant, bright lights in the sky**.

C++에서 샘플링만 greedy(k=1)로 바꾼 경우:

> Based on the image, it appears the person is looking towards the right side of the frame.

Python BF16의 첫 문장(누락된 첫 토큰을 복원):

> Based on the image, the skateboarder is looking **up and slightly to his right**.

C++ greedy + 비전 양방향 마스크:

> Based on the image, the skateboarder is looking slightly upwards and to his right.

## 1. 비전 attention 마스크 오류 — 통제 실험으로 확인

`vai/nodes/attention.h`에서 `causalBits = 1`로 초기화된다.
`VisionTower`는 encoder에 대해 `updateCausality(0)`을 호출하지 않는다.
`kvCacheSize = 0`은 KV 캐시만 끄며 causal masking을 끄지 않는다.
실제 shader도 causalBits에 따라 query t가 볼 수 있는 범위를 t+1까지 제한한다.
따라서 SigLIP의 4096개 이미지 패치가 전체 패치를 보는 대신 자기 앞의 패치만 본다.
설치된 Transformers의 SiglipAttention은 명시적으로 `is_causal = False`다.

임시 실행 파일에서 다른 연산 조건을 유지하고 아래 한 줄만 추가했다.

```cpp
model.visionTower->encoder.updateCausality(0);
```

PyTorch FP32 기준 상대 RMS 오차(100 × RMSE/reference RMS):

| 지점 | 기존 C++ FP16 | 비전 양방향 설정 후 |
|---|---:|---:|
| vis.embed | 0.0682% | 0.0682% |
| vis.00 | 9.9158% | 0.1538% |
| vis.proj | 94.4634% | 4.9647% |
| txt.embed | 92.7295% | 4.8735% |
| prefill 마지막 위치 logits | 18.9373% | 1.2235% |

기존 구현은 같은 이력에서도 6번째 생성 토큰(index 5, step.04.logits)에서 갈라진다.

| 조건 | ` the` logit | ` it` logit | 선택 |
|---|---:|---:|---|
| 기존 C++ FP16 | 40.3767 | 42.7274 | ` it` |
| Python BF16 | 54.2500 | 52.0000 | ` the` |
| Python FP32 | 53.6632 | 51.1419 | ` the` |
| C++ 비전 양방향 | 54.2982 | 51.9941 | ` the` |

다음 위치에서 기존 C++는 ` person`, Python은 ` skateboard`를 선택한다.
비전 양방향 설정 후 C++도 ` skateboard`를 선택한다.
FP32 기준으로 확인한 prefill + 16개 teacher-forced decode의 17개 argmax가 모두 일치했다.
이는 자유 생성 전체나 모든 입력에 대한 일치 보장은 아니다.

수정 위치는 공용 Transformer가 아니라 Gemma3 `VisionTower` 생성자다.
`Classic`을 사용하는 다른 모델은 causal attention을 필요로 할 수 있다.

```cpp
// models/gemma3/model.h — VisionTower constructor body
encoder.updateCausality(0);  // SigLIP vision encoder is bidirectional.
```

## 2. 샘플링 정책 차이

`TopKSampling::Config` 기본값은 k=40, temperature=1이다.
C++ demo는 이를 바꾸지 않고, 매 forward에서 std::rand() 기반 uniform 값을 공급한다.
워밍업도 난수열을 소비한다. Python은 매번 logits.argmax()를 사용한다.

따라서 수치가 같아도 기본 C++와 Python 문장은 같아야 할 이유가 없다.
비교용 C++에서는 weights 로딩 후, 워밍업 전에 다음 설정을 사용했다.

```cpp
model.topkSampling.config_.k = 1;
```

## 3. 남는 수치 차이와 autoregressive 분기

비전 마스크를 고쳐도 C++ FP16과 Python BF16은 비트 단위로 같지 않다.
가중치/KV/activation dtype, attention 내부 정밀도, CUDA/Vulkan 연산 경로가 다르다.
BF16 실행은 입력 pixel_values도 모델 실행 전에 BF16으로 바꾼다.
저장된 pixel_values.npy가 float32라고 실제 추론까지 FP32인 것은 아니다.

수정된 C++와 Python BF16의 첫 argmax 차이는 11번째 생성 토큰(index 10, step.09)이다.
Python BF16은 ` **`와 ` slightly`가 40.25로 동점이며 작은 token ID의 ` **`를 선택한다.
Python FP32는 ` slightly`를 선택하고, 두 후보의 차이는 약 0.08793이다.
수정된 C++도 ` slightly`를 선택한다. 여기서부터 자유 생성의 문장과 KV 이력이 달라진다.

BF16 기준 65개 선택 중 수정 전 C++는 14개, 수정 후는 4개가 다르다.
다만 이 카운트에는 Python 종료 토큰 이후의 무의미한 생성도 포함된다.
마지막 행에서 RMS 오차가 작아지는 것만으로 해당 레이어가 오차를 교정했다고 해석하면 안 된다.
특히 C++ 마지막 decoder layer는 마지막 토큰 행만 출력한다.

## 4. Python 최종 출력 처리

`demo.py:156`:

```python
text_out = processor.decode(fed[1:] + [picks[-1]], skip_special_tokens=True)
```

`fed[0]`은 prefill이 예측한 첫 생성 토큰 22515(`Based`)이다.
이를 제외해서 실제 화면 출력이 ` on the image...`로 시작한다.
전체 생성열은 `fed + [picks[-1]]`이다. 답변 표시에는 종료 토큰까지만 사용해야 한다.

또한 Python loop는 `<eos>`/`<end_of_turn>`을 검사하지 않고 64번 실행한다.
이번 BF16 실행은 생성 index 42에서 `<end_of_turn>`이 나왔지만 계속 생성하여 끝에 `)` 등이 붙었다.
`skip_special_tokens=True`는 종료 이후 일반 토큰까지 제거하지 않는다.
C++는 종료 토큰을 검사하며, 최대 길이도 128 decode steps로 다르다.
덤프 목적의 고정 step 유지가 필요하면 생성 덤프와 사용자에게 표시할 답변을 분리해야 한다.

## 전처리와 JPEG 차이의 위치

이전 검증에서 동일 RGB에 대한 preprocessImage는 processor와 정확히 일치했다.
STB/Pillow JPEG 디코딩 차이는 실제 실행에 남지만, 이번 마스크 비교에서는
동일 pixel_values.npy를 공유하여 배제했다. 따라서 큰 오차의 원인을 JPEG로 설명할 수 없다.
최종 자유 생성 A/B도 같은 C++ JPEG 디코더를 사용하면서 마스크만 변경했다.

## 산출물

모든 실행 로그와 덤프: `tests/gemma3/dumps/final-analysis/`

- `demo-cpp-fp16.log`: 기본 C++
- `demo-cpp-greedy-fp16.log`: greedy만 변경
- `demo-python-bf16.log`, `torch-bf16/`: Python 기본 실행
- `vai-fp16-shared/`: 기존 C++ / 동일 픽셀 / teacher forcing
- `torch-fp32-forced/`: CPU FP32 / 동일 토큰 이력 16 steps
- `vai-bidirectional/`: 비전 양방향 / 동일 픽셀 / teacher forcing
- `demo-cpp-bidirectional.log`: greedy + 비전 양방향 자유 생성
- `metrics.txt`, `metrics-bidirectional.txt`: 레이어 및 토큰별 수치

주의: `metrics.txt`의 bf16-fp32 logits 통계는 전체 prefill 행을 포함한다.
`metrics-bidirectional.txt`의 logits 통계는 모든 쌍에서 마지막 위치만 사용한다.
본문의 logits 수치는 항상 마지막 위치 기준이다.

기존 dump-vai.cpp는 modelCfg.numLayers 참조 때문에 현재 API로 빌드되지 않았다.
진단용 임시 복사본에서만 model.decoder.layers.size()로 치환했다.
기존 데모 및 모델 구현은 이번 분석 과정에서 수정하지 않았다.
