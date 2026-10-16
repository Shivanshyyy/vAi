# vAi

**DAG 기반 Vulkan 추론 엔진.** 학습된 신경망을 연산 노드의 그래프로 조립하고
Vulkan compute shader 위에서 실행합니다. Vulkan conformant GPU라면 벤더와 무관하게
같은 코드가 돕니다.

<!-- TODO: 데모 GIF (llama3 토큰 스트리밍 / yolov8 검출 결과) — docs/assets/ 에 둘 것 -->

```cpp
struct Model : public vai::NeuralNet {
    ConvSequence convX2;
    vai::FlattenNode flatten;
    vai::LinearNode linear;

    Model(const vai::NeuralNetConfig& cfg)
    : NeuralNet(cfg), convX2({1, 32, 64}, 3), linear({7 * 7 * 64, 10})
    {
        input("in0") - convX2 - flatten - linear - output("out0");
    }
};
```

```cpp
Tensor output = model({{"in0", input}})["out0"];
```

## 특징

- **벤더 무관** — NVIDIA · AMD · Intel 어디서든 같은 SPIR-V. CUDA도 벤더 런타임도 필요 없음
- **모델 포맷 없음** — 그래프가 곧 C++ 코드. 변환 도구를 거치지 않음
- **커널 다중 경로** — 형상과 장치에 따라 협조 행렬 / 스칼라 / 폴백 커널을 자동 선택
- **KV 캐시 융합 어텐션** — 캐시 append와 causal attention이 한 커널. GQA · 슬라이딩 윈도우 · split-K 디코드 지원
- **SafeTensors 직접 로드** — HuggingFace 체크포인트를 그대로 읽음
- **빌트인 노드 27종** — [레퍼런스](docs/ko/06-nodes.md)

## 제공 모델

| 모델 | 종류 | 문서 |
|---|---|---|
| MNIST | 이미지 분류 | [빠른 실행](docs/ko/02-quickstart.md) |
| YOLOv8 | 객체 검출 | [yolov8.md](docs/ko/04-models/yolov8.md) |
| YOLOX | 객체 검출 | [yolox.md](docs/ko/04-models/yolox.md) |
| GPT-2 124M | 언어 모델 | [gpt2.md](docs/ko/04-models/gpt2.md) |
| Llama 3.2 1B | 언어 모델 | [llama3.md](docs/ko/04-models/llama3.md) |

## 빌드

Vulkan SDK, CMake 3.16+, C++23 컴파일러가 필요합니다.

```bash
git clone --recursive https://github.com/<owner>/vAi.git
cd vAi
cmake -B build -DVAI_BUILD_MODEL_TESTS=ON
cmake --build build --config RelWithDebInfo
./build/bin/test-model-mnist
```

체크포인트는 저장소에 없습니다. 자세한 내용은
[빠른 실행](docs/ko/02-quickstart.md) 참조.

## 문서

| 장 | 내용 |
|---|---|
| [개요](docs/ko/index.md) | 특징, 프로젝트 구조 |
| [빠른 실행](docs/ko/02-quickstart.md) | 설치, 최소 예제 |
| [vAi 사용법](docs/ko/03-usage/README.md) | 노드 · 노드그룹 · NeuralNet · Tensor |
| [제공 모델](docs/ko/04-models/README.md) | 모델별 설명과 성능 |
| [커스텀 모델 작성](docs/ko/05-contributing-models.md) | 새 모델 올리기 |
| [제공 노드들](docs/ko/06-nodes.md) | 빌트인 노드 레퍼런스 |
| [커스텀 노드와 커널 작성](docs/ko/07-contributing-nodes.md) | 새 연산 추가하기 |

## 성능

> 측정값 미기입 — [통합 벤치마크](docs/ko/04-models/README.md#통합-벤치마크) 참조.

## 라이선스

[MIT](LICENSE)
