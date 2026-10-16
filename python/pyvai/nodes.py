"""Concrete graph node classes for building vAi models.

The C extension exposes nodes as `IdentityNode`, `LinearNode`, ... here we
re-export them under the `pyvai.nodes` namespace so they can be addressed
together:

    from pyvai import nodes

    class Net(pyvai.NeuralNet):
        def __init__(self):
            super().__init__()
            self.fc = nodes.LinearNode(in_dim=128, out_dim=64)

Base classes (`Node`, `NodeGroup`, ...), state objects (`KVCacheCounter`),
and config enums (`Epilogue`) stay at the top-level `pyvai` namespace.

Adding a builtin node touches five sites — keep them in sync:
  1. src/bindings/nodes/bind-vai-<node>.cpp   (the `bindXxx` definition)
  2. src/bindings/nodes/bind-nodes.h          (its declaration)
  3. src/bindings/nodes/bind-nodes.cpp        (the `bindNodes` call, ordered)
  4. this file's `from ._pyvai import (...)`   (re-export)
  5. this file's `__all__`
"""
from ._pyvai import (
    # shape / plumbing
    IdentityNode,
    GateNode,
    ReshapeNode,
    FlattenNode,
    ConcatNode,
    PermuteNode,
    SliceNode,
    TypeCastNode,
    # element-wise
    AddNode,
    MulNode,
    GatedMulNode,
    ActivationNode,
    # vision
    ConvolutionNode,
    MaxPool2dNode,
    AvgPool2dNode,
    NearestUpsampleNode,
    # linear / normalization
    LinearNode,
    LayerNormNode,
    RmsNormNode,
    BatchNormNode,
    # transformer
    TokenEmbeddingNode,
    RopeNode,
    AttentionNode,
    # output heads
    ArgmaxNode,
    TopKSamplingNode,
)


__all__ = [
    "IdentityNode",
    "GateNode",
    "ReshapeNode",
    "FlattenNode",
    "ConcatNode",
    "PermuteNode",
    "SliceNode",
    "TypeCastNode",
    "AddNode",
    "MulNode",
    "GatedMulNode",
    "ActivationNode",
    "ConvolutionNode",
    "MaxPool2dNode",
    "AvgPool2dNode",
    "NearestUpsampleNode",
    "LinearNode",
    "LayerNormNode",
    "RmsNormNode",
    "BatchNormNode",
    "TokenEmbeddingNode",
    "RopeNode",
    "AttentionNode",
    "ArgmaxNode",
    "TopKSamplingNode",
]
