"""NeuralNet + graph DSL: wiring (both styles), routing, and the lifetime model.

The core keeps raw NodeSlot* links and assumes nodes outlive the graph; the
binding mirrors that with net-anchored adoption (bind-base-node.cpp). The
lifetime tests here were use-after-free crashes before that machinery.
"""
import gc
import threading

import numpy as np
import pytest

import pyvai
from pyvai import nodes


def _run(net, n=4):
    x = pyvai.Tensor.from_numpy(np.arange(n, dtype=np.float32))
    return net({"in0": x})["out0"].to_numpy()


# ---------------------------------------------------------------------------
# Forward basics
# ---------------------------------------------------------------------------

def test_call_rejects_non_tensor():
    net = _identity_net()
    with pytest.raises(TypeError):
        net({"in0": np.arange(4, dtype=np.float32)})


def test_slot_indexing_unknown_slot_raises():
    with pytest.raises(ValueError, match="no such slot"):
        _ = nodes.IdentityNode()["nonexistent"]


def test_wiring_unknown_slot_lists_real_slots():
    """The `-` chain used to surface a bare map::at; now it names the slots."""
    net = pyvai.NeuralNet()
    with pytest.raises(ValueError, match=r"no such slot 'out0'.*then"):
        net.input("in0") - nodes.GateNode() - net.output("out0")


def test_rtruediv_on_flow_routes_in_slot():
    """'in1' / (a - b): __rtruediv__ existed on Node/NodeGroup but not NodeFlow."""
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
    out = Net()({"a": pyvai.Tensor.from_numpy(a),
                 "b": pyvai.Tensor.from_numpy(b)})["out0"].to_numpy()
    np.testing.assert_array_equal(out, a + b)


# ---------------------------------------------------------------------------
# `-` DSL: slot routing and groups
# ---------------------------------------------------------------------------

def test_slot_routing_into_two_input_node():
    """`"in0"/node` and `"in1"/node` route into named input slots of a 2-arg node."""
    class Net(pyvai.NeuralNet):
        def __init__(self):
            super().__init__()
            self.add = nodes.AddNode()
            self.input("in0") - "in0"/self.add - self.output("out0")
            self.input("in1") - "in1"/self.add

    a = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    b = np.array([10.0, 20.0, 30.0], dtype=np.float32)
    out = Net()({"in0": pyvai.Tensor.from_numpy(a),
                 "in1": pyvai.Tensor.from_numpy(b)})["out0"].to_numpy()
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

    arr = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    out = Net()({"in0": pyvai.Tensor.from_numpy(arr)})["out0"].to_numpy()
    np.testing.assert_array_equal(arr, out)


# ---------------------------------------------------------------------------
# Call wiring: node(x) is sugar over x - node
# ---------------------------------------------------------------------------

def test_call_positional_args_go_to_in0_in1():
    net = pyvai.NeuralNet()
    a = nodes.IdentityNode()(net.input("a"))
    b = nodes.IdentityNode()(net.input("b"))
    net.output("out0")(nodes.AddNode()(a, b))

    x = np.arange(4, dtype=np.float32)
    y = np.full(4, 2.0, dtype=np.float32)
    out = net({"a": pyvai.Tensor.from_numpy(x),
               "b": pyvai.Tensor.from_numpy(y)})["out0"].to_numpy()
    np.testing.assert_allclose(out, x + y)


def test_call_keyword_args_go_to_named_slots():
    net = pyvai.NeuralNet()
    a = nodes.IdentityNode()(net.input("a"))
    b = nodes.IdentityNode()(net.input("b"))
    net.output("out0")(nodes.MulNode()(a, in1=b))

    x = np.arange(1, 5, dtype=np.float32)
    y = np.full(4, 3.0, dtype=np.float32)
    out = net({"a": pyvai.Tensor.from_numpy(x),
               "b": pyvai.Tensor.from_numpy(y)})["out0"].to_numpy()
    np.testing.assert_allclose(out, x * y)


# ---------------------------------------------------------------------------
# Lifetime model — temporaries wired into a net are owned by the net
# ---------------------------------------------------------------------------

def test_temporary_nodes_in_chain_survive():
    """Nothing holds the middle nodes but the net."""
    net = pyvai.NeuralNet()
    net.input("in0") - nodes.IdentityNode() - nodes.IdentityNode() - net.output("out0")
    gc.collect()
    np.testing.assert_array_equal(_run(net), np.arange(4, dtype=np.float32))


def test_slot_spec_flow_keeps_temporary_alive():
    """`node / "name"` flows drag their wrapper refs into the adoption."""
    net = pyvai.NeuralNet()
    net.input("in0") - (nodes.IdentityNode() / "out0") - net.output("out0")
    gc.collect()
    np.testing.assert_array_equal(_run(net), np.arange(4, dtype=np.float32))


def test_define_slot_pins_the_slot_argument():
    """define_slot must pin the slot arg (keep_alive<1,3> — not <1,2>, the
    name). Observed via a sentinel in the node's __dict__: deterministic,
    unlike the forward test below, which a lenient heap can let pass."""
    import weakref

    class Sentinel:
        pass

    g = pyvai.NodeGroup()
    n = nodes.IdentityNode()
    s = Sentinel()
    n._sentinel = s
    w = weakref.ref(s)
    g.define_slot("in0", n.slot("in0"))
    del n, s
    gc.collect()
    assert w() is not None      # the group still holds the node wrapper


def test_define_slot_keeps_local_node_alive():
    """define_slot's keep_alive chain: group -> slot wrapper -> node."""
    class G(pyvai.NodeGroup):
        def __init__(self):
            super().__init__()
            ident = nodes.IdentityNode()        # local: dies with __init__ unless kept
            self.define_slot("in0", ident.slot("in0"))
            self.define_slot("out0", ident.slot("out0"))

    net = pyvai.NeuralNet()
    net.input("in0") - G() - net.output("out0")  # the group itself is a temporary too
    gc.collect()
    np.testing.assert_array_equal(_run(net), np.arange(4, dtype=np.float32))


def test_boundary_nodes_keep_net_alive():
    """input()/output() wrappers hold the net via _pyvai_net."""
    inp = pyvai.NeuralNet().input("in0")
    gc.collect()
    assert isinstance(inp, pyvai.InputNode)     # net not collected under the wrapper


def test_attribute_held_nodes_still_work():
    """The documented-safe subclass pattern is unaffected."""
    net = _identity_net()
    gc.collect()
    np.testing.assert_array_equal(_run(net), np.arange(4, dtype=np.float32))


def _identity_net():
    class Net(pyvai.NeuralNet):
        def __init__(self):
            super().__init__()
            self.id = nodes.IdentityNode()
            self.input("in0") - self.id - self.output("out0")
    return Net()

def test_node_ctor_without_net_raises():
    """vai::Node latches the thread-local current net; null latch was a segfault.

    A worker thread never saw a NeuralNet, so its latch is deterministically
    null whatever ran earlier in this process."""
    caught = []

    def build():
        try:
            nodes.AddNode()
        except RuntimeError as e:
            caught.append(str(e))

    t = threading.Thread(target=build)
    t.start()
    t.join()
    assert caught and "no active NeuralNet" in caught[0]
