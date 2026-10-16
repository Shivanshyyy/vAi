"""Builtin nodes — one numerical reference test per kernel family.

Deliberately not exhaustive: each test covers the family's binding path
(config, slots, dtype resolution) plus the kernel against a numpy reference.
Nodes sharing a family's plumbing (e.g. Mul next to Add) are not repeated.
"""
import gc

import numpy as np

import pyvai
from pyvai import nodes


# ---------------------------------------------------------------------------
# Shape / plumbing
# ---------------------------------------------------------------------------

def test_reshape_node_changes_shape_preserves_data():
    arr = np.arange(12, dtype=np.float32)

    class Net(pyvai.NeuralNet):
        def __init__(self):
            super().__init__()
            self.r = nodes.ReshapeNode([3, 4])
            self.input("in0") - self.r - self.output("out0")

    out = Net()({"in0": pyvai.Tensor.from_numpy(arr)})["out0"]
    assert list(out.shape) == [3, 4]
    np.testing.assert_array_equal(out.to_numpy(), arr.reshape(3, 4))


# ---------------------------------------------------------------------------
# Element-wise
# ---------------------------------------------------------------------------

def test_add_node_elementwise():
    class Net(pyvai.NeuralNet):
        def __init__(self):
            super().__init__()
            self.add = nodes.AddNode()
            self.input("in0") - "in0"/self.add - self.output("out0")
            self.input("in1") - "in1"/self.add

    a = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    b = np.array([10.0, 20.0, 30.0, 40.0], dtype=np.float32)
    out = Net()({"in0": pyvai.Tensor.from_numpy(a),
                 "in1": pyvai.Tensor.from_numpy(b)})["out0"].to_numpy()
    np.testing.assert_array_equal(out, a + b)


def test_gated_mul_silu_gates_the_value_half(rng):
    """GatedMulNode: out[..., j] = silu(in[..., j]) * in[..., H + j]."""
    H = 4

    class Net(pyvai.NeuralNet):
        def __init__(self):
            super().__init__()
            self.g = nodes.GatedMulNode(activation=pyvai.Epilogue.silu)
            self.input("in0") - self.g - self.output("out0")

    x = rng.standard_normal((2, 2 * H)).astype(np.float32)
    out = Net()({"in0": pyvai.Tensor.from_numpy(x)})["out0"].to_numpy()

    gate, value = x[:, :H], x[:, H:]
    expected = gate / (1.0 + np.exp(-gate)) * value
    assert out.shape == (2, H)
    np.testing.assert_allclose(out, expected, atol=1e-4)


# ---------------------------------------------------------------------------
# Vision
# ---------------------------------------------------------------------------

def test_conv_bn_silu_matches_reference():
    """ConvolutionNode(has_batch_norm=True) = Conv + BN(γ,β,μ,σ²) + activation fused."""
    class Net(pyvai.NeuralNet):
        def __init__(self):
            super().__init__(w_default_dtype=pyvai.Dtype.F32)
            self.c = nodes.ConvolutionNode(
                in_channels=1, out_channels=1, kernel_size=1, stride=1,
                has_bias=False, has_batch_norm=True,
                epilogue=pyvai.Epilogue.silu,
            )
            self.input("in0") - self.c - self.output("out0")

    net = Net()
    # 1x1 identity conv; BN: scale by 2, shift by 0 (μ=0, σ²=1 → bn(x) ≈ 2x).
    net.c["weight"]   = pyvai.Tensor.from_numpy(np.array([[[[1.0]]]], dtype=np.float32))
    net.c["gamma"]    = pyvai.Tensor.from_numpy(np.array([2.0],       dtype=np.float32))
    net.c["beta"]     = pyvai.Tensor.from_numpy(np.array([0.0],       dtype=np.float32))
    net.c["mean"]     = pyvai.Tensor.from_numpy(np.array([0.0],       dtype=np.float32))
    net.c["variance"] = pyvai.Tensor.from_numpy(np.array([1.0],       dtype=np.float32))

    arr = np.array([[[-1.0, 2.0], [3.0, -0.5]]], dtype=np.float32)  # (Ci=1, H=2, W=2)
    out = net({"in0": pyvai.Tensor.from_numpy(arr)})["out0"].to_numpy()

    bn   = 2.0 * arr
    silu = bn / (1.0 + np.exp(-bn))
    np.testing.assert_allclose(out, silu, atol=1e-4)


def test_nearest_upsample_node_replicates_pixels():
    class Net(pyvai.NeuralNet):
        def __init__(self):
            super().__init__()
            self.u = nodes.NearestUpsampleNode(scale=2)
            self.input("in0") - self.u - self.output("out0")

    arr = np.array([[[1.0, 2.0], [3.0, 4.0]]], dtype=np.float32)  # (C=1, H=2, W=2)
    out = Net()({"in0": pyvai.Tensor.from_numpy(arr)})["out0"].to_numpy()
    expected = np.repeat(np.repeat(arr, 2, axis=1), 2, axis=2)    # (C=1, H=4, W=4)
    np.testing.assert_array_equal(out, expected)


# ---------------------------------------------------------------------------
# Linear / normalization
# ---------------------------------------------------------------------------

def test_linear_with_bias():
    """Identity weight + bias: out = x + b."""
    class Net(pyvai.NeuralNet):
        def __init__(self):
            super().__init__(w_default_dtype=pyvai.Dtype.F32)
            self.l = nodes.LinearNode(in_dim=4, out_dim=4, has_bias=True)
            self.input("in0") - self.l - self.output("out0")

    net = Net()
    net.l["weight"] = pyvai.Tensor.from_numpy(np.eye(4, dtype=np.float32))
    net.l["bias"]   = pyvai.Tensor.from_numpy(np.array([10.0, 20.0, 30.0, 40.0], dtype=np.float32))

    arr = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    out = net({"in0": pyvai.Tensor.from_numpy(arr)})["out0"]
    np.testing.assert_allclose(out.to_numpy(), np.array([11.0, 22.0, 33.0, 44.0], dtype=np.float32))


def test_rms_norm_matches_reference(rng):
    """RmsNormNode: x / sqrt(mean(x^2) + eps) * scale over the feature axis."""
    D, eps = 16, 1e-5

    class Net(pyvai.NeuralNet):
        def __init__(self):
            super().__init__(w_default_dtype=pyvai.Dtype.F32)
            self.n = nodes.RmsNormNode(eps=eps)
            self.input("in0") - self.n - self.output("out0")

    scale = (rng.standard_normal(D) * 0.5 + 1.0).astype(np.float32)
    net = Net()
    net.n["scale"] = pyvai.Tensor.from_numpy(scale)

    x = (rng.standard_normal((4, D)) * 3).astype(np.float32)
    out = net({"in0": pyvai.Tensor.from_numpy(x)})["out0"].to_numpy()

    expected = x / np.sqrt((x ** 2).mean(axis=-1, keepdims=True) + eps) * scale
    np.testing.assert_allclose(out, expected, atol=1e-3)


# ---------------------------------------------------------------------------
# Transformer
# ---------------------------------------------------------------------------

def test_attention_node_causal_sdpa(rng):
    """AttentionNode = softmax(QK^T / sqrt(D)) V with causal mask, out [N, H, D]."""
    H, N, D = 2, 4, 8

    class Net(pyvai.NeuralNet):
        def __init__(self):
            super().__init__()
            # kv_cache_size defaults to 0: no cache, so each forward stands alone.
            self.attn = nodes.AttentionNode(num_q_heads=H, head_dim=D,
                                            causal=True, kv_cache_dtype=pyvai.Dtype.F32)
            self.input("in0") - "Q"/self.attn - self.output("out0")
            self.input("in1") - "K"/self.attn
            self.input("in2") - "V"/self.attn

    q = rng.standard_normal((H, N, D)).astype(np.float32)
    k = rng.standard_normal((H, N, D)).astype(np.float32)
    v = rng.standard_normal((H, N, D)).astype(np.float32)
    out = Net()({"in0": pyvai.Tensor.from_numpy(q),
                 "in1": pyvai.Tensor.from_numpy(k),
                 "in2": pyvai.Tensor.from_numpy(v)})["out0"].to_numpy()

    scale  = 1.0 / np.sqrt(D)
    scores = np.einsum("hqd,hkd->hqk", q, k) * scale
    mask   = np.triu(np.full((N, N), -1e30, dtype=np.float32), k=1)
    scores = scores + mask
    weights = np.exp(scores - scores.max(axis=-1, keepdims=True))
    weights = weights / weights.sum(axis=-1, keepdims=True)
    expected = np.einsum("hqk,hkd->hqd", weights, v)

    # out0 is token-major [N, H, D], so the head axis moves to compare.
    np.testing.assert_allclose(out, expected.transpose(1, 0, 2), atol=1e-3)


def test_rope_position_zero_is_identity(rng):
    """RopeNode at position 0 rotates by angle 0 (identity); at a later position
    it changes the vector but, being a rotation, preserves each token's norm.
    Positions come from the KVCacheCounter."""
    H, N, Dh = 2, 1, 8

    counter = pyvai.KVCacheCounter()

    class Net(pyvai.NeuralNet):
        def __init__(self):
            super().__init__()
            self.r = nodes.RopeNode(head_dim=Dh)
            self.r.set_kv_cache_counter(counter)
            self.input("in0") - self.r - self.output("out0")

    x = rng.standard_normal((H, N, Dh)).astype(np.float32)
    net = Net()

    out0 = net({"in0": pyvai.Tensor.from_numpy(x)})["out0"].to_numpy()
    np.testing.assert_allclose(out0, x, atol=1e-5)

    counter.update(7)
    out7 = net({"in0": pyvai.Tensor.from_numpy(x)})["out0"].to_numpy()
    assert not np.allclose(out7, x, atol=1e-3)          # actually rotated
    np.testing.assert_allclose(                          # rotations preserve norms
        np.linalg.norm(out7, axis=-1), np.linalg.norm(x, axis=-1), atol=1e-3)


# ---------------------------------------------------------------------------
# Output heads
# ---------------------------------------------------------------------------

def test_topk_sampling_k1_is_argmax(rng):
    """TopKSamplingNode with k=1: the single surviving candidate is the row max,
    so sampling is deterministic whatever the uniform is."""
    class Net(pyvai.NeuralNet):
        def __init__(self):
            super().__init__()
            self.t = nodes.TopKSamplingNode(k=1)
            self.input("in0") - self.t - self.output("out0")

    x = rng.standard_normal((2, 33)).astype(np.float32)
    net = Net()
    net.t.set_uniform(0.5)
    out = net({"in0": pyvai.Tensor.from_numpy(x)})["out0"]
    got = out.get_data("uint32").reshape(-1)
    np.testing.assert_array_equal(got, np.argmax(x, axis=1).astype(np.uint32))


# ---------------------------------------------------------------------------
# KV-cache state — the counter the cached nodes read positions from
# ---------------------------------------------------------------------------

def test_kv_cache_counter_state():
    """KVCacheCounter wraps a uint32 buffer with update / reset / read."""
    counter = pyvai.KVCacheCounter()
    assert counter.value == 0
    counter.update(42)
    assert counter.value == 42
    counter.reset()
    assert counter.value == 0


def test_counter_survives_python_reference_via_node():
    """set_kv_cache_counter pins the counter to its node (keep_alive):
    the counter's dtor destroys the Vulkan buffer, so without the pin this
    forward would read freed memory once the local reference is collected."""
    class Net(pyvai.NeuralNet):
        def __init__(self):
            super().__init__()
            self.r = nodes.RopeNode(head_dim=8)
            c = pyvai.KVCacheCounter()
            c.update(7)
            self.r.set_kv_cache_counter(c)   # keep_alive<1, 2>
            self.input("in0") - self.r - self.output("out0")

    net = Net()
    gc.collect()                                 # the local `c` dies here
    x = np.ones((2, 1, 8), dtype=np.float32)
    out = net({"in0": pyvai.Tensor.from_numpy(x)})["out0"].to_numpy()
    assert not np.allclose(out, x, atol=1e-3)    # position 7: actually rotated


# ---------------------------------------------------------------------------
# Control flow
# ---------------------------------------------------------------------------

def test_gate_node_records_only_the_open_branch():
    class Net(pyvai.NeuralNet):
        def __init__(self):
            super().__init__()
            self.gate  = nodes.GateNode()
            self.relu  = nodes.ActivationNode(epilogue=pyvai.Epilogue.relu)
            self.ident = nodes.IdentityNode()
            out = self.output("out0")           # both branches merge here
            self.input("in0") - self.gate
            (self.gate / "then") - self.relu - out
            (self.gate / "else") - self.ident - out

    arr = np.array([-2.0, -1.0, 0.0, 1.0, 2.0], dtype=np.float32)
    net = Net()
    assert net.gate.open                        # gates open by default

    out = net({"in0": pyvai.Tensor.from_numpy(arr)})["out0"].to_numpy()
    np.testing.assert_array_equal(out, np.maximum(arr, 0.0))

    net.gate.open = False                       # flips the branch, re-records
    out = net({"in0": pyvai.Tensor.from_numpy(arr)})["out0"].to_numpy()
    np.testing.assert_array_equal(out, arr)


# ---------------------------------------------------------------------------
# Profiling
# ---------------------------------------------------------------------------

def test_profile_tag_can_change_after_the_first_forward():
    """The core re-records on a tag change, so tagging is not init-only."""
    class Net(pyvai.NeuralNet):
        def __init__(self):
            super().__init__()
            self.id = nodes.IdentityNode()
            self.input("in0") - self.id - self.output("out0")

    net = Net()
    x = lambda: pyvai.Tensor.from_numpy(np.arange(4, dtype=np.float32))
    net({"in0": x()})["out0"].to_numpy()

    net.id.profile_tag = "late"                 # used to abort the interpreter
    assert net.id.profile_tag == "late"
    net({"in0": x()})["out0"].to_numpy()
    assert net.gpu_time_ms("late") >= 0.0

    net.id.profile_tag = ""                     # empty tag untags
    assert net.id.profile_tag == ""
