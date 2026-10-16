# slab mat-vec 커널 A/B

`linear-slab`(M ≤ 8의 mat-vec 경로)에서 GLSL을 어떻게 쓰느냐가 시간에 얼마나 영향을 주는지
가르기 위한 셰이더 묶음과 그 측정 결과. 알고리즘·그리드·옮기는 바이트는 전부 동일하고, 표기만
다른 파일들이다.

문서 규격은 [docs/optimizations/](../../../docs/optimizations/)의 것을 따른다.


## Test Target

**Module**: `LinearNode`의 슬랩 경로 (`src/shaders/linear-slab.comp`)
- 용도: 디코드의 Q/K/V·proj·FC1·FC2·lm_head
- 입력: `O[M, N] = I[M, K] · Wᵀ`, M = 1, W는 fp16, I/O는 fp32
- 호출 빈도: 토큰당 레이어 수 × 4~6회 + lm_head 1회

**Source Files**
| 파일 | 역할 |
|---|---|
| [test-linear-slab.comp](test-linear-slab.comp) | 기준선. 출하 커널에서 비교 축만 남김 |
| [test-linear-slab-v5.comp](test-linear-slab-v5.comp) | 기준선 + 루프 수정 3개. fold 유지 |
| [test-linear-slab-v2.comp](test-linear-slab-v2.comp) | v5 + 서브그룹 전용 stride. fold 유지 |
| [test-linear-slab-v0.comp](test-linear-slab-v0.comp) | v2 − `shared` 선언과 fold |
| [test-linear-slab2.comp](test-linear-slab2.comp) | 계보가 다른 서브그룹 전용 커널. 참조 |
| [test-stream.comp](test-stream.comp) | 읽기 대역폭 루프라인 |
| [test-nodes.h](test-nodes.h) | 각 셰이더의 노드. 상속 없이 분리 |
| [main.cpp](main.cpp) | 정확성 |
| [main2.cpp](main2.cpp) | 성능 |

**Build & Run**
```bash
cmake --build build/RelWithDebInfo --target test-linear-slab

build/RelWithDebInfo/bin/test-linear-slab            # 형상별 성능
build/RelWithDebInfo/bin/test-linear-slab --stats    # 드라이버의 컴파일 통계
```

`--stats`는 별도 프로세스에서만 돌린다. 통계용 파이프라인을 만들었다 지우는 것만으로 대조군이
열 배로 벌어진다.

**Reference**: `llama.cpp/ggml/src/ggml-vulkan/vulkan-shaders/mul_mat_vec.comp`


## Environment

- **GPU**: Intel(R) Arc(TM) Graphics (내장)
- **OS**: Windows 11
- **Driver**: 101.8424 (Vulkan 0.406.232)
- **Vulkan**: API 1.4.333
- **Subgroup**: 기본 32, 범위 [8, 32]. 파이프라인이 16으로 고정(`ggmlSlabSgSize`)


## 파일별 차이

기준선에 대해 무엇이 얹혔는지. ①②③이 A/B가 찾아낸 루프 수정이고, ④⑤⑥은 그 위에 얹은 제거다.

| | ① W 호이스트 | ② 2-deep 제거 | ③ 레인을 서브그룹에서 | ④ 서브그룹 전용 stride | ⑤ fold·`shared` 제거 | ⑥ `K_PER_ITER` 상수화 |
|---|---|---|---|---|---|---|
| slab | – | – | – | – | – | – |
| **v5** | ✓ | ✓ | ✓ | – | – | – |
| **v2** | ✓ | ✓ | ✓ | ✓ | – | ✓ |
| **v0** | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |

| | 내용 |
|---|---|
| ① | W를 누적 루프 안에서 읽지 않고 `vec4 wv[wgN]`로 먼저 끌어올림 |
| ② | 4-deep과 tail 사이의 2-deep 언롤 단계 삭제 |
| ③ | `gl_LocalInvocationID.x` → `gl_SubgroupID * sgSize + gl_SubgroupInvocationID` |
| ④ | 레인을 `gl_SubgroupInvocationID` 단독으로, `numIters`/`k0`의 stride를 `wgSize` → `sgSize` |
| ⑤ | `shared float tmpsh[...]` 선언과 `if (wgSize > sgSize)` fold 블록 삭제 |
| ⑥ | `K_PER_ITER`를 스펙 상수에서 평범한 상수 4로, `K_PER_ITER / 4` 래퍼 루프 삭제 |

**유효 범위가 갈린다.** ④는 레인이 K를 `sgSize`씩 건너뛰게 만들므로, 워크그룹이 더 넓으면 모든
서브그룹이 같은 슬랩을 중복해서 걷는다. 그래서 v2·v0은 `wgSize == sgSize`에서만 유효하고
(`SlabV2Node`/`SlabV0Node`가 assert), 긴 K가 쓰는 4-서브그룹 기하는 slab과 v5만 표현할 수 있다.


## 컴파일 통계

`VK_KHR_pipeline_executable_properties`로 드라이버에게 직접 물은 값. 전부 정적 수치다.

| | slab | v5 | v2 | v0 | slab2 |
|---|---|---|---|---|---|
| Instruction Count | 1589 | 2196 | 2190 | 2190 | 2192 |
| Basic Block Count | 132 | 147 | 147 | 147 | 147 |
| Scratch Memory Size | 0 | 0 | 0 | 0 | 0 |
| Loop Count | 10 | 8 | 8 | 8 | 8 |
| Workgroup Memory Size | 8 | 8 | **8** | **0** | 0 |

- **어느 커널도 스필하지 않는다** (`Scratch = 0`). 레지스터 초과 가설은 여기서 끝난다.
- ①②③이 명령어를 607개 늘리고 루프를 둘 줄인다. 언롤이 실제로 펼쳐졌다는 뜻이고, **코드가 큰
  쪽이 빠르다** — 정적 명령어 수는 속도 지표가 아니다.
- ④는 명령어 6개, ⑤는 SLM 8바이트, ⑥은 **아무것도** 바꾸지 않는다.
- **v2와 v0의 컴파일 차이는 SLM 8바이트가 전부다.** 다섯 항목 중 나머지 넷이 동일하다.
- 드라이버가 내주는 통계는 이 다섯 개뿐이다. 레지스터 수도 SIMD 폭도 없고,
  `internalRepresentations()`는 0개를 반환한다(상용 드라이버가 ISA를 안 연다).


## 측정 결과

`p25 of 45 rounds`, 독립 duel 5회의 중앙값, 프로세스 3회. `wg`는 휴리스틱이 slab·v5에 준 값이고
v2·v0은 항상 `sg`로 고정된다. 그래서 **`wg == sg`인 행만 순수한 셰이더 비교**다.

### wg = sg = 16

| 형상 | v5 vs slab | v2 vs slab | v0 vs slab |
|---|---|---|---|
| **768×768** | -0.4 / -1.1 / +0.5 | +2.6 / -0.6 / +3.9 | **+8.5 / +3.9 / +7.3** |
| 768×3072 | +0.4 / +1.1 / +1.1 | -0.1 / +1.1 / +0.8 | -0.8 / +0.0 / -1.7 |
| 768×50257 | +0.0 / -0.2 / -0.4 | -0.0 / -0.2 / -0.3 | -1.7 / -1.3 / -1.7 |
| 2048×128256 | +1.6 / -0.4 / -0.6 | -2.3 / -0.3 / -0.5 | +3.1 / +0.5 / +0.6 |

### wg = 64 (v2·v0은 16으로 강제 — 기하 + 셰이더)

| 형상 | v5 vs slab | v2 vs slab | v0 vs slab |
|---|---|---|---|
| **2048×512** | +1.1 / +2.2 / +0.7 | -1.7 / +0.5 / -2.2 | **+4.8 / +4.4 / +5.5** |
| 3072×768 | +0.9 / +2.4 / +1.6 | -0.2 / -0.5 / -0.9 | +2.2 / +0.2 / +1.0 |
| 2048×2048 | +0.5 / +0.5 / +0.5 | -1.9 / +0.2 / -2.0 | -1.5 / +0.2 / -0.5 |
| **8192×2048** | -0.4 / +0.4 / +0.4 | -2.5 / -3.1 / -2.0 | **-6.6 / -6.2 / -6.2** |
| 2048×8192 | +0.9 / +0.2 / +0.3 | +1.5 / +1.0 / +1.4 | +2.0 / +1.7 / +1.5 |

대조군은 이 실행들에서 0.3~2.3%였다(`2048×128256` 한 행만 6.9%).

### `shared` 선언 하나의 값 (v0 − v2)

두 파일은 그것만 다르다.

| 형상 | run1 | run2 | run3 |
|---|---|---|---|
| 768×768 | +5.9 | +4.5 | +3.4 |
| 2048×512 | +6.5 | +3.9 | +7.7 |
| **8192×2048** | **-4.1** | **-3.1** | **-4.2** |

**방향이 형상마다 뒤집힌다.** 작은 형상에서는 없는 편이, 긴 K에서는 있는 편이 빠르다. 단순한
점유율 세금이라면 한 방향이어야 한다.


## 판정

| # | 수정 | 결과 | 비고 |
|---|---|---|---|
| 1 | ①②③ (= v5) | **NO EFFECT** | 9행 전부 ±2.4%, 대부분 대조군 이하 |
| 2 | ①②③ 중 하나만 | **NO EFFECT** | 셋 다 있어야 언롤이 펼쳐진다 |
| 3 | ④ 추가 (= v2) | **NO EFFECT** | 명령어 6개, 판정도 노이즈 |
| 4 | ⑤ 추가 (= v0) | **+4~8% / -6%** | 작은 형상에서 이득, 긴 K에서 손해 |
| 5 | ⑥ | **NO EFFECT** | 컴파일 결과가 v0과 완전히 동일 |

**출하 커널을 바꿀 근거는 아직 없다.** v5는 이득이 없고, v0의 이득은 `768×768`과 `2048×512`
두 행에 국한되며 `8192×2048`에서 6%를 잃는다. 두 셰이더를 형상으로 갈라 쓰려면 v0의 불안정성
(Known Issues #1)을 먼저 설명해야 한다.


## Failed Patterns (DO NOT RETRY)

| 패턴 | 결과 | 왜 |
|---|---|---|
| W를 `f16vec4`로 들고 `dot` 직전에 확장 | 기댓값 +3.2% → +0.9% | 레지스터를 아낀 만큼 확장 연산이 의존 사슬 위로 올라온다. 이봉은 사라지지만 이득도 사라진다 |
| `wgN`을 2 → 1로 낮춰 레지스터 여유 확보 | 전부 느려짐, v0의 이득 소멸 | W 행이 하나면 묶어서 발행할 로드가 없다. 게다가 입력 벡터를 워크그룹 수만큼 더 읽는다 |
| ①②③ 중 하나만 적용 | NO EFFECT × 3 | 서로를 막는다 |
| ⑥(`K_PER_ITER` 상수화)만 기대 | NO EFFECT | 스펙 상수는 이미 완전히 접힌다 |
| 전처리기로 순열 만들기 | 컨벤션 위반 | 이 저장소는 순열을 특수화 상수로만 낸다. 전략이 다르면 파일을 나눈다 |
| 커널 내부 최적화로 작은 형상 개선 | 근거 없음 | `test-stream`(순수 읽기)이 작은 형상에서 slab보다 **느리다**. 연산을 전부 없애도 안 빨라지므로 커널 안에 여유가 없다 |


## Known Issues

### Issue 1: v0의 시간이 프로세스마다 두 상태로 갈린다
- **증상**: `768×768`에서 p25가 0.0283~0.0305 ms를 오간다. 한 프로세스 안에서는 5회 duel이
  ±0.6%로 붙지만, 프로세스가 바뀌면 7%가 움직인다
- **재현 조건**: `wgN = 2`, `wgSize == sgSize == 16`. v2·slab2는 같은 조건에서 훨씬 안정적
- **현재 상태**: 🔴 미해결
- **추정 원인**: 드라이버가 GRF 모드(128 ↔ 256)를 프로세스마다 다르게 고를 가능성. 256을 고르면
  XVE당 상주 스레드가 절반이 된다. **확인 불가** — 이 드라이버가 레지스터 수를 안 준다
- **배제된 것**: 스필(`Scratch = 0`), `OpSpecConstantOp` 미접힘(미리 접은 `.spv`를 주입해도 무변),
  harness 편향(같은 `.spv`를 두 슬롯에 넣으면 전 행 0.0 ± 0.6%)

### Issue 2: 엔트리 구성이 다른 엔트리의 값을 바꾼다
- **증상**: `duel`의 엔트리 하나를 교체하면 손대지 않은 엔트리의 판정도 몇 %p 움직인다
- **현재 상태**: 🟡 조사중
- **메모**: 한 실행 안의 비교만 유효하다고 보고, 서로 다른 실행의 표를 가로질러 읽지 말 것.
  절대 시간도 세션에 따라 8%까지 밀린다(장치 상태)

### Issue 3: `2048×512`에서 휴리스틱이 LARGE를 고른다
- **증상**: `wgSize`를 64 → 16으로 내린 v0이 3회 모두 +4.4~+5.5%
- **현재 상태**: 🟡 조사중
- **메모**: [ggmlSlabWgSize](../../../vai/nodes/linear.h)의 `N <= 8192 && K >= 1024`는 ggml이
  양자화 커널 기준으로 정한 문턱이라 이 장치에 맞을 근거가 원래 없다. `K = 2048`에서는 서브그룹
  하나가 슬랩을 통째로 맡는 편이 낫고 `K = 8192`에서는 아니다. **셰이더를 하나도 안 건드리고
  호스트 조건만 바꿔 잴 수 있는 유일한 후보**


## Related Documents

- [docs/slab-kernel-shapes.md](../../../docs/slab-kernel-shapes.md) — 형상별 작업 분해와 실효 대역폭
- [docs/vai-linear-vs-ggml-mulmat.md](../../../docs/vai-linear-vs-ggml-mulmat.md) — ggml 원본과의 차이


## Next Steps

- [ ] `main.cpp`를 CMake에서 살려 v0·v2·v5를 `unaligned-K` 포함해 통과시킨다 — 지금 어느 변형도
      정확성 검증을 안 거쳤다
- [ ] Issue 3을 stock slab만으로 잰다 (호스트 한 줄)
- [ ] Issue 1은 리눅스 Mesa/ANV에서 ISA와 레지스터 통계를 뽑아 확인한다

---

*Keywords: mat-vec, GEMV, slab, subgroup, shared memory, GRF, occupancy, Intel Arc, specialization constant*
