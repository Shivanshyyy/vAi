# NodeGroup

A NodeGroup is a subgraph that groups multiple nodes to be used as one. It is used to group repeated structures like Transformer blocks into one unit.

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

1. Declare nodes as members,
2. Connect the internals in the constructor,
3. Decide which slots to expose externally.


`NodeGroup::defineSlot` does not create a new slot but assigns an external name to an internal node's slot. If connected to the group's `in0` from outside, it actually connects to the `in0` slot of `fc1`.



NodeGroup does not have `prepare()` or `run()`. The nodes inside are executed, and the group is just a name for grouping them.

## Connection

Like nodes, you can connect them with `-`. You can also mix groups and nodes.

```cpp
norm1 - mhAttn - add1;      // Node - Group - Node
ffn1 - ffn2;                // Group - Group
```

The rules for specifying slots using `/` are the same. If it is in front of the group, it is the input slot, and if it is behind the group, it is the output slot. The names that can be used are exposed by `defineSlot`. Likewise, as with nodes, if `/` is omitted, it is regarded as `in0` and `out0`.

## Nesting

You can put another group inside a group. Llama 3.2's Transformer block is a good example. `MultiHeadAttention` and `GatedFeedForward` are groups respectively, and `TransformerBlock`, which contains these two, is also a group.

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

The two lines of wiring represent the data flow itself. The second line is a residual connection.

## Identity
In the example above, `input` is an `Identity` that does not perform any calculations. When an input splits into multiple paths, it creates a branching point. It is used to send the same input to two different places, like in a residual connection.

## Precautions when using containers

Nodes and NodeGroups can be placed in the container. However, elements placed in the container must stay at the place where they were created if they are connected by wiring using `-`, as their slots hold each other's addresses.

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

`std::vector` does not automatically preserve that condition. Whenever elements are added, elements may be recreated and copied inside as needed. If the space is insufficient, they are moved to a larger memory as a whole, and if a value created outside is added, its copy is stored. The wiring remains pointing to the original address regardless of the direction. The compilation passes, but it causes a runtime error when running the graph.

It is the user's responsibility to keep the elements in their original places without moving, and there are two possible ways.

**1. `reserve` + `emplace_back`**: It prevents reallocation by reserving space in advance for the required number of elements and prevents copying by creating the elements directly in the container. The above example uses this method.

**2. `std::unique_ptr`**: It puts the elements in the heap, and the container only holds pointers. Even if the vector moves the pointer, the object it points to does not move.

```cpp
std::vector<std::unique_ptr<TransformerBlock>> blocks;
blocks.push_back(std::make_unique<TransformerBlock>(...));
```

vAi allows Method 2 but encourages users to use Method 1.

Other STL containers, including `std::deque`, `std::list`, and `std::map`, also require the contained elements to stay at the place where they were created. Which operations move elements varies by container, but that is a standard C++ rule, so we do not cover it here.
