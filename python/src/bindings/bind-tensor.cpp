#include "pyvai-internal.h"

#include "vai/engine/tensor.h"

#include <cstring>
#include <memory>
#include <string>
#include <vector>


namespace nb = nanobind;

namespace pyvai {

namespace {

using NpFloat = nb::ndarray<float, nb::c_contig, nb::device::cpu>;
using NpAny = nb::ndarray<nb::c_contig, nb::device::cpu>;


template <typename Arr>
std::vector<uint32_t> shapeFromNdarray(const Arr& array)
{
    std::vector<uint32_t> shape(array.ndim());
    for (size_t i = 0; i < array.ndim(); ++i)
        shape[i] = static_cast<uint32_t>(array.shape(i));
    return shape;
}


// Byte width of a nanobind/DLPack dtype (e.g. numpy int32 -> 4).
size_t itemsizeOf(const nb::dlpack::dtype& dt)
{
    return static_cast<size_t>(dt.bits) / 8 * dt.lanes;
}


// Resolve a numpy dtype name (e.g. "uint32", "float16") to the DLPack dtype
// numpy expects plus its byte width. Used by get_data for the raw typed view.
struct NpDtypeInfo { nb::dlpack::dtype dt; size_t itemsize; };

NpDtypeInfo resolveNpDtype(const std::string& name)
{
    using Code = nb::dlpack::dtype_code;
    auto make = [](Code code, uint8_t bits) -> NpDtypeInfo {
        return { nb::dlpack::dtype{ static_cast<uint8_t>(code), bits, 1 },
                 static_cast<size_t>(bits) / 8 };
    };

    if (name == "float32" || name == "f4") return make(Code::Float, 32);
    if (name == "float16" || name == "f2") return make(Code::Float, 16);
    if (name == "int32"   || name == "i4") return make(Code::Int,   32);
    if (name == "uint32"  || name == "u4") return make(Code::UInt,  32);
    if (name == "int16"   || name == "i2") return make(Code::Int,   16);
    if (name == "uint16"  || name == "u2") return make(Code::UInt,  16);
    if (name == "int8"    || name == "i1") return make(Code::Int,    8);
    if (name == "uint8"   || name == "u1") return make(Code::UInt,   8);

    throw nb::value_error(
        ("get_data: unsupported dtype '" + name + "' "
         "(use float32/float16/int32/uint32/int16/uint16/int8/uint8)").c_str());
}


// Raw byte copy of an untyped numpy array into `dst` (no dtype conversion).
// `dst` must already have a shape; the array's itemsize must equal the tensor
// element size and element counts must match. Mirrors vai::Tensor::setData<T>.
void setRawData(vai::Tensor& dst, const NpAny& array, const char* where)
{
    if (!dst.validShape())
        throw nb::value_error(
            (std::string(where) + ": tensor must have a shape").c_str());
    if (dst.baseOffset() != 0)   // core setData asserts (aborts) on offset views
        throw nb::value_error(
            (std::string(where) + ": not supported on an offset view").c_str());

    const size_t itemsize = itemsizeOf(array.dtype());
    if (itemsize != vai::bytesPerElement(dst.dtype()))
        throw nb::value_error(
            (std::string(where) + ": array itemsize must equal the tensor "
             "element size (raw copy, no conversion)").c_str());
    if (array.size() != dst.numElements())
        throw nb::value_error(
            (std::string(where) + ": array size does not match tensor shape").c_str());

    const void* p = array.data();
    switch (itemsize)
    {
        case 1: dst.setData(static_cast<const uint8_t*>(p));  break;
        case 2: dst.setData(static_cast<const uint16_t*>(p)); break;
        case 4: dst.setData(static_cast<const uint32_t*>(p)); break;
        default:
            throw nb::value_error(
                (std::string(where) + ": unsupported element size").c_str());
    }
}


// Readback guard — turns the device-readback path that aborts in the core into
// a Python RuntimeError, using marks NeuralNet.__call__ stamps on the exact
// wrappers it returns:
//   outputs: _pyvai_net / _pyvai_out_gen / _pyvai_read (the next forward
//            reuses the output buffer, staling unread older outputs)
// Unmarked wrappers (reshape()/permute() views, other sources) fall through
// unchanged — the guard fails open, never blocking a servable read.
void guardDeviceReadback(nb::handle self, const vai::Tensor& t)
{
    if (t.hasHostData() || !t.hasDeviceData())
        return;                                     // host path, or error path downstream

    if (nb::hasattr(self, "_pyvai_out_gen"))        // marked forward output
    {
        if (nb::cast<bool>(self.attr("_pyvai_read")))
            return;                                 // re-read: core serves the synced cache
        auto mine = nb::cast<uint64_t>(self.attr("_pyvai_out_gen"));
        auto cur  = nb::cast<uint64_t>(self.attr("_pyvai_net").attr("_pyvai_forward_gen"));
        if (mine != cur)
            throw std::runtime_error(
                "reading a stale forward output: a later forward on this net "
                "reused its buffer. Read each output before the next call "
                "(or keep to_numpy() copies).");
        self.attr("_pyvai_read") = true;
    }
}


// Shared to_numpy / get_data tail: host the bytes (GIL dropped for the fence
// wait), copy into a capsule-owned buffer (unique_ptr until handover, so no
// throw leaks). outBytes must equal the tensor's byte size — raw bytes only.
nb::ndarray<nb::numpy> readbackToNdarray(vai::Tensor& t, size_t outBytes,
                                         nb::dlpack::dtype npDtype,
                                         const char* who)
{
    auto buf = std::make_unique<std::vector<char>>(outBytes);

    const void* src = nullptr;
    if (t.hasHostData())
    {
        src = t.hostData();
    }
    else if (t.hasDeviceData())
    {
        // Forward outputs are lazy device views: toCpu() waits the forward's
        // fence and copies device -> host before we read it. That wait can be
        // long, so let other Python threads run while it blocks.
        nb::gil_scoped_release rel;
        src = t.toCpu().hostData();
    }
    else
    {
        throw nb::value_error(
            (std::string(who) + ": tensor has neither host nor device data").c_str());
    }

    std::memcpy(buf->data(), src, outBytes);

    const auto& shape = t.shape();
    std::vector<size_t> npShape(shape.begin(), shape.end());

    nb::capsule owner(buf.get(), [](void* p) noexcept {
        delete static_cast<std::vector<char>*>(p);
    });
    std::vector<char>* raw = buf.release();   // the capsule owns it now

    return nb::ndarray<nb::numpy>(
        raw->data(),
        npShape.size(),
        npShape.data(),
        owner,
        /*strides*/ nullptr,
        npDtype
    );
}


// The output buffer is always freshly allocated and owned by the returned
// ndarray's capsule.
nb::ndarray<nb::numpy> tensorToNumpy(vai::Tensor t)
{
    if (!t.validShape())
        throw nb::value_error("to_numpy: tensor has no shape");
    if (t.dtype() == vai::Dtype::None)
        throw nb::value_error("to_numpy: tensor has no dtype");

    // Build the DLPack dtype that nanobind will hand to numpy. C++ has no
    // portable fp16 scalar type so the fp16 dtype is constructed by hand.
    nb::dlpack::dtype npDtype;
    switch (t.dtype())
    {
        case vai::Dtype::Fp32:
            npDtype = nb::dtype<float>();
            break;
        case vai::Dtype::Fp16:
            npDtype = nb::dlpack::dtype{
                static_cast<uint8_t>(nb::dlpack::dtype_code::Float), 16, 1
            };
            break;
        default:
            // Unreachable: Dtype::None was rejected above.
            throw nb::value_error("to_numpy: internal error, unhandled dtype");
    }

    const size_t outBytes = t.numElements() * vai::bytesPerElement(t.dtype());
    return readbackToNdarray(t, outBytes, npDtype, "to_numpy");
}

}   // namespace


void bindTensor(nb::module_& m)
{
    // BF16 is a safetensors file format only, not a tensor dtype: the parser
    // widens it to fp32 on load, so it never surfaces here. NONE is what an
    // empty (default-constructed) tensor reports; it is not a creatable dtype,
    // so the creation paths below reject it.
    nb::enum_<vai::Dtype>(m, "Dtype")
        .value("F32",  vai::Dtype::Fp32)
        .value("F16",  vai::Dtype::Fp16)
        .value("NONE", vai::Dtype::None);

    nb::class_<vai::Tensor>(m, "Tensor", nb::dynamic_attr())
        .def(nb::init<>())
        .def(
            "__init__",
            [](vai::Tensor* self, const std::vector<uint32_t>& shape, vai::Dtype dtype) {
                // The core TensorData ctor asserts (aborts) on Dtype::None.
                if (dtype == vai::Dtype::None)
                    throw nb::value_error("Tensor: dtype must be F32 or F16, not NONE");
                new (self) vai::Tensor(shape, dtype);
            },
            nb::arg("shape"),
            nb::arg("dtype") = vai::Dtype::Fp32
        )

        .def_static(
            "from_numpy",
            [](NpFloat array, vai::Dtype dtype) -> vai::Tensor {
                if (dtype == vai::Dtype::None)
                    throw nb::value_error("from_numpy: dtype must be F32 or F16, not NONE");
                vai::Tensor t(shapeFromNdarray(array), dtype);
                t.setFloats(array.data());
                return t;
            },
            nb::arg("array"),
            nb::arg("dtype") = vai::Dtype::Fp32,
            "Create a Tensor from a numpy.ndarray. The numpy array must be "
            "F32 / C-contiguous; the tensor stores it converted to `dtype`."
        )

        .def_static(
            "from_data",
            [](NpAny array, vai::Dtype dtype) -> vai::Tensor {
                if (dtype == vai::Dtype::None)
                    throw nb::value_error("from_data: dtype must be F32 or F16, not NONE");
                vai::Tensor t(shapeFromNdarray(array), dtype);
                setRawData(t, array, "from_data");
                return t;
            },
            nb::arg("array"),
            nb::arg("dtype") = vai::Dtype::Fp32,
            "Create a Tensor by raw-copying `array`'s bytes into a `dtype`-typed "
            "slot — NO dtype conversion (mirror of the C++ Tensor::setData<T>). "
            "The array's itemsize must equal bytesPerElement(dtype); use this for "
            "integer payloads such as token ids stored in an F32-sized slot."
        )

        .def_prop_ro("shape", [](const vai::Tensor& t) {
            const auto& s = t.shape();
            return std::vector<uint32_t>(s.begin(), s.end());
        })
        .def_prop_ro("dtype",        &vai::Tensor::dtype)
        .def_prop_ro("num_elements", &vai::Tensor::numElements)
        .def_prop_ro("num_bytes",    &vai::Tensor::numBytes)

        .def(
            "set",
            [](vai::Tensor& self, NpFloat array) -> vai::Tensor& {
                if (!self.validShape())
                    throw nb::value_error("set: tensor must have a shape (use Tensor(shape) or from_numpy)");
                if (self.baseOffset() != 0)   // core setFloats asserts (aborts) on offset views
                    throw nb::value_error("set: not supported on an offset view");
                if (self.numElements() != array.size())
                    throw nb::value_error("set: numpy size does not match tensor shape");
                self.setFloats(array.data());
                return self;
            },
            nb::arg("array"),
            nb::rv_policy::reference,
            "In-place copy F32 numpy data into this tensor's host buffer. "
            "The data is converted to the tensor's stored dtype."
        )

        .def(
            "set_data",
            [](vai::Tensor& self, NpAny array) -> vai::Tensor& {
                setRawData(self, array, "set_data");
                return self;
            },
            nb::arg("array"),
            nb::rv_policy::reference,
            "In-place raw byte copy of `array` into this tensor's host buffer — "
            "NO dtype conversion (mirror of the C++ Tensor::setData<T>). The "
            "array's itemsize must equal the tensor element size."
        )

        .def(
            "reshape",
            [](const vai::Tensor& self, std::vector<uint32_t> dims) -> vai::Tensor {
                if (!self.hasHostData())
                    throw nb::value_error("reshape: requires host data (call set() or use from_numpy)");
                size_t total = 1;
                for (uint32_t d : dims) total *= d;
                if (self.numElements() != total)
                    throw nb::value_error("reshape: total element count must remain constant");
                return self.reshape(dims);
            },
            nb::arg("shape"),
            "The same data seen as `shape` — a sharing view; this tensor is "
            "left untouched (mirror of the C++ Tensor::reshape)."
        )

        .def(
            "permute",
            [](const vai::Tensor& self, nb::args perms) -> vai::Tensor {
                if (!self.hasHostData())
                    throw nb::value_error("permute: requires host data (call set() or use from_numpy)");

                std::vector<uint32_t> p;
                p.reserve(perms.size());
                for (auto h : perms)
                    p.push_back(nb::cast<uint32_t>(h));

                const uint32_t rank = self.shape().size();
                if (p.size() != rank)
                    throw nb::value_error("permute: arg count must match tensor rank");
                for (uint32_t v : p)
                    if (v >= rank)
                        throw nb::value_error("permute: index out of range");
                for (size_t i = 0; i < p.size(); ++i)
                    for (size_t j = i + 1; j < p.size(); ++j)
                        if (p[i] == p[j])
                            throw nb::value_error("permute: duplicate index");

                // vai::Tensor::permute is a variadic template; dispatch by rank.
                switch (rank)
                {
                    case 1: return self.permute(p[0]);
                    case 2: return self.permute(p[0], p[1]);
                    case 3: return self.permute(p[0], p[1], p[2]);
                    case 4: return self.permute(p[0], p[1], p[2], p[3]);
                    default:
                        throw nb::value_error("permute: rank > 4 is not supported");
                }
            },
            "Return a NEW tensor with the axes reordered (mirror of the C++ "
            "Tensor::permute, which is const). NOT in-place: the original is "
            "left untouched."
        )

        .def("to_numpy",
            [](nb::handle_t<vai::Tensor> self) {
                const auto& t = nb::cast<const vai::Tensor&>(self);
                guardDeviceReadback(self, t);
                return tensorToNumpy(t);
            },
            "Copy this tensor into a new numpy array (native dtype). The read "
            "that aborts in C++ raises RuntimeError here instead: a forward "
            "OUTPUT never read before the net's next call reused its buffer. The guard tracks the exact objects passed "
            "through __call__ — reading the same storage via another wrapper "
            "(e.g. a reshape() view) bypasses it and can still abort.")

        .def(
            "get_data",
            [](nb::handle_t<vai::Tensor> self, const std::string& dtype) -> nb::ndarray<nb::numpy> {
                auto& t = nb::cast<vai::Tensor&>(self);
                guardDeviceReadback(self, t);
                if (!t.validShape())
                    throw nb::value_error("get_data: tensor has no shape");

                NpDtypeInfo info = resolveNpDtype(dtype);
                if (info.itemsize != vai::bytesPerElement(t.dtype()))
                    throw nb::value_error(
                        "get_data: requested dtype itemsize must equal the tensor "
                        "element size (raw view, no conversion)");

                const size_t bytes = t.numElements() * info.itemsize;
                return readbackToNdarray(t, bytes, info.dt, "get_data");
            },
            nb::arg("dtype"),
            "Raw typed view of the tensor's bytes as numpy `dtype` — NO "
            "conversion (mirror of the C++ Tensor::getData<T>). Waits the "
            "producing forward and reads back device data when needed. The "
            "dtype itemsize must equal the tensor element size. "
            "Same readback guard as to_numpy(): stale outputs raise "
            "RuntimeError; unmarked wrappers of the same "
            "storage (views) can still hit the core abort."
        )

        .def("__repr__", [](const vai::Tensor& t) -> std::string {
            std::string s = "Tensor(shape=[";
            if (t.validShape())
            {
                const auto& sh = t.shape();
                for (size_t i = 0; i < sh.size(); ++i)
                {
                    if (i) s += ", ";
                    s += std::to_string(sh[i]);
                }
            }
            s += "], dtype=";
            switch (t.dtype())
            {
                case vai::Dtype::Fp32: s += "F32";  break;
                case vai::Dtype::Fp16: s += "F16";  break;
                case vai::Dtype::None: s += "none"; break;
            }
            s += ")";
            return s;
        });
}

}   // namespace pyvai