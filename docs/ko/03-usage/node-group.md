# 노드그룹(NodeGroup)

`NodeGroup`은 노드 여러 개를 감싸 하나처럼 쓰는 서브그래프입니다. 트랜스포머
블록처럼 반복되는 구조를 한 단위로 묶을 때 씁니다.

```cpp
struct FeedForward : public vai::NodeGroup {
    vai::node::Linear fc1, fc2;                 // 1

    FeedForward(uint32_t in, uint32_t hidden, uint32_t out, vai::Epilogue act)
    : fc1({in, hidden, true, vai::Epilogue::GeluTanh})
    , fc2({hidden, out, true})
    {
        fc1 - fc2;                            // 2

        defineSlot("in0", fc1.slot("in0"));   // 3
        defineSlot("out0", fc2.slot("out0"));
    }
};
```

1. 멤버로 노드를 선언하고,
2. 생성자에서 내부를 잇고,
3. 밖에 노출할 슬롯을 정합니다.


`NodeGroup::defineSlot`은 새 슬롯을 만드는 것이 아니라 내부 노드의 슬롯에 외부 이름을
붙이는 것입니다. 밖에서 그룹의 `in0`에 연결하면 실제로는 `fc1`의 `in0` 슬롯에
잇습니다.



노드그룹에는 `prepare()`도 `run()`도 없습니다. 실행되는 것은 안에 든 노드들이고,
그룹은 그것들을 묶어 부르는 이름일 뿐입니다.

## 연결

노드와 똑같이 `-`로 잇습니다. 그룹과 노드를 섞어도 됩니다.

```cpp
norm1 - mhAttn - add1;      // 노드 - 그룹 - 노드
ffn1 - ffn2;                // 그룹 - 그룹
```

`/`로 슬롯을 지정하는 규칙도 같습니다. 그룹 앞이면 들어가는 슬롯, 뒤면 나가는
슬롯이고, 쓸 수 있는 이름은 `defineSlot`으로 노출한 것뿐입니다. 노드와 마찬가지로
`/`를 생략하면 `in0`과 `out0`으로 간주합니다.

## 중첩

그룹 안에 다른 그룹을 둘 수도 있습니다. Llama 3.2의 트랜스포머 블록이 좋은
예인데, `MultiHeadAttention`과 `GatedFeedForward`가 각각 하나의 그룹이고, 이
둘을 담은 `TransformerBlock` 역시 그룹입니다.

```cpp
struct TransformerBlock : public vai::NodeGroup {
    vai::node::Identity input;
    vai::node::RmsNorm norm1;
    MultiHeadAttention mhAttn;
    vai::node::Add add1;
    vai::node::RmsNorm norm2;
    GatedFeedForward ffn;
    vai::node::Add add2;

    TransformerBlock(...)
    {
        input - norm1 - mhAttn - "in0"/add1 - norm2 - ffn - "in0"/add2;
        input                  - "in1"/add1               - "in1"/add2;

        defineSlot("in0", input.slot("in0"));
        defineSlot("out0", add2.slot("out0"));
    }
};
```

두 줄의 배선이 곧 데이터 흐름입니다. 둘째 줄이 잔차 연결입니다.

## Identity
위 예에서 `input`은 아무 연산도 하지 않는 `Identity`입니다. 입력이 여러
갈래로 갈라질 때 그 분기점을 만들어 줍니다. 잔차 연결처럼 같은 입력을 두 곳으로
보내야 할 때 씁니다.

## 컨테이너 사용 시 주의할 점

노드도 노드그룹도 컨테이너에 담을 수 있습니다. 다만 `-`로 이은 배선은 슬롯이
서로의 주소를 들고 있어서 담긴 원소는 만들어진 자리에 그대로 머물러야
합니다.

```cpp
struct TransformerSequence : public vai::NodeGroup {
    std::vector<TransformerBlock> blocks;

    TransformerSequence(uint32_t numLayers, ...)
    {
        blocks.reserve(numLayers);
        for (uint32_t i = 0; i < numLayers; ++i)
        {
            blocks.emplace_back(...);
            if (i > 0)
                blocks[i - 1] - blocks[i];
        }

        defineSlot("in0", blocks.front().slot("in0"));
        defineSlot("out0", blocks.back().slot("out0"));
    }
};
```

`std::vector`는 그 조건을 저절로 지켜 주지 않습니다. 원소를 넣을 때마다 필요에
따라 안에서 원소를 다시 만들고 복사합니다. 자리가 모자라면 더 큰 메모리로 통째로
옮기고, 밖에서 만든 값을 넣으면 그 복사본을 담습니다. 어느 쪽이든 배선은 원래
주소를 가리킨 채 남습니다. 컴파일은 통과하고 그래프를 실행할 때 터집니다.

원소를 제자리에 묶어 두는 것은 작성자 몫이고, 방법은 두 가지입니다.

**1. `reserve` + `emplace_back`**: 필요한 개수만큼 자리를 미리 잡아 재할당을
막고 컨테이너 안에서 원소를 직접 만들어 복사를 막습니다. 위 예가 이 방식입니다.

**2. `std::unique_ptr`**: 원소를 힙에 두고 컨테이너는 포인터만 담습니다. 벡터가
포인터를 옮겨도 가리키는 객체는 움직이지 않습니다.

```cpp
std::vector<std::unique_ptr<TransformerBlock>> blocks;
blocks.push_back(std::make_unique<TransformerBlock>(...));
```

vAi는 2번 방식도 허용하지만 1번을 권장합니다.

`std::deque`, `std::list`, `std::map` 등 다른 STL 컨테이너도 마찬가지로 담긴
원소가 만들어진 자리에 머물러야 합니다. 어느 연산에서 원소가 옮겨지는지는
컨테이너마다 다르지만 그것은 표준 C++ 쪽 규칙이라 여기서는 다루지 않습니다.
