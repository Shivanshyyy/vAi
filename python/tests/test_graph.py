"""NeuralNet and the graph DSL: wiring, routing and lifetimes.

The core links raw NodeSlot*, so a wired node must outlive its net; the
lifetime test covers the binding's net-anchored adoption (bindings/engine/graph-flow.cpp).
"""
import gc

import numpy as np

import pyvai
from pyvai import nodes


def _run(net, n=4):
    x = pyvai.Tensor.from_numpy(np.arange(n, dtype=np.float32))
    return net(x).to_numpy()


def test_dsl_slot_routing():
    """`"in1"/node` routes into a named input slot, on a node and on a flow."""
    class Net(pyvai.NeuralNet):
        def __init__(self):
            super().__init__()
            self.add = nodes.AddNode()
            i0 = nodes.IdentityNode()
            self.input("a") - i0
            self.input("b") - "in1" / (i0 - self.add)
            self.add - self.output("out0")

    a = np.array([1.0, 2.0], dtype=np.float32)
    b = np.array([10.0, 20.0], dtype=np.float32)
    out = Net()(a=pyvai.Tensor.from_numpy(a), b=pyvai.Tensor.from_numpy(b)).to_numpy()
    np.testing.assert_array_equal(out, a + b)


def test_node_group_subclass_in_neural_net():
    """A user-defined NodeGroup subclass plugs into a NeuralNet via define_slot."""
    class TwoIdentity(pyvai.NodeGroup):
        def __init__(self):
            super().__init__()
            self.id1 = nodes.IdentityNode()
            self.id2 = nodes.IdentityNode()
            self.id1 - self.id2
            self.define_slot("in0",  self.id1.slot("in0"))
            self.define_slot("out0", self.id2.slot("out0"))

    class Net(pyvai.NeuralNet):
        def __init__(self):
            super().__init__()
            self.group = TwoIdentity()
            self.input("in0") - self.group - self.output("out0")

    np.testing.assert_array_equal(_run(Net()), np.arange(4, dtype=np.float32))


def test_temporaries_wired_into_net_survive():
    """Nothing holds the middle nodes but the net — plain and via a
    `node / "name"` slot-spec flow."""
    net = pyvai.NeuralNet()
    net.input("in0") - nodes.IdentityNode() - (nodes.IdentityNode() / "out0") - net.output("out0")
    gc.collect()
    np.testing.assert_array_equal(_run(net), np.arange(4, dtype=np.float32))
