#include "pyvai-internal.h"

#include "vai/engine/safetensors-parser.h"
#include "vai/engine/tensor.h"

#include <optional>
#include <stdexcept>


namespace nb = nanobind;

namespace pyvai {

namespace {

void ensureKeyExists(const vai::SafetensorRef& ref, const std::string& key)
{
    try
    {
        (void) ref.getDataType();
    }
    catch (const std::out_of_range&)
    {
        PyErr_SetString(PyExc_KeyError, key.c_str());
        throw nb::python_error();
    }
}

// dtypeFromString (tensor.h) VAI_ASSERTs on anything else — debug builds
// abort, release builds silently mislabel the bytes as fp32.
void ensureLoadableDtype(const vai::SafetensorRef& ref, const std::string& key)
{
    const std::string dt = ref.getDataType();
    if (dt != "F32" && dt != "F16" && dt != "BF16")
    {
        PyErr_Format(PyExc_TypeError,
                     "'%s' has dtype %s; pyvai loads F32 / F16 / BF16 only "
                     "(ref('%s') still inspects it without loading)",
                     key.c_str(), dt.c_str(), key.c_str());
        throw nb::python_error();
    }
}

}   // namespace


void bindSafetensors(nb::module_& m)
{
    // Metadata-only handle: shape/dtype without loading the tensor data. A ref
    // holds a raw parser pointer, so parser.ref() ties it to its parser with
    // keep_alive.
    nb::class_<vai::SafetensorRef>(m, "SafetensorRef")
        .def_prop_ro(
            "key",
            [](const vai::SafetensorRef& self) { return self.key; }
        )
        .def_prop_ro(
            "shape",
            [](const vai::SafetensorRef& self) { return self.getShape(); },
            "Tensor shape from the file header — no data is read."
        )
        .def_prop_ro(
            "dtype",
            [](const vai::SafetensorRef& self) { return self.getDataType(); },
            "The file's dtype tag (e.g. 'F32', 'F16', 'BF16') — the on-disk "
            "format, which may not be a loadable vai Dtype (BF16 converts at "
            "load time)."
        )
        .def(
            "__repr__",
            [](const vai::SafetensorRef& self) -> std::string {
                std::string s = "SafetensorRef('" + self.key + "', " +
                                self.getDataType() + ", [";
                auto shape = self.getShape();
                for (size_t i = 0; i < shape.size(); ++i)
                {
                    if (i) s += ", ";
                    s += std::to_string(shape[i]);
                }
                return s + "])";
            }
        );

    nb::class_<vai::SafetensorsParser>(m, "SafetensorsParser")
        .def(
            "__init__",
            [](vai::SafetensorsParser* self, const std::string& path) {
                ensureFileExists("SafetensorsParser", path);
                new (self) vai::SafetensorsParser(path.c_str());
            },
            nb::arg("path")
        )

        .def("tensor_names", &vai::SafetensorsParser::getTensorNames)

        .def(
            "__getitem__",
            [](const vai::SafetensorsParser& self, const std::string& key) -> vai::Tensor {
                auto ref = self[key];
                ensureKeyExists(ref, key);
                ensureLoadableDtype(ref, key);
                // Disk read + convert, pure C++ — drop the GIL.
                nb::gil_scoped_release rel;
                return vai::Tensor(ref);
            },
            nb::arg("key"),
            "Load `key` as a Tensor in the safetensor file's native dtype."
        )

        .def(
            "get",
            [](const vai::SafetensorsParser& self,
               const std::string& key,
               std::optional<vai::Dtype> dtype) -> vai::Tensor {
                auto ref = self[key];
                ensureKeyExists(ref, key);
                ensureLoadableDtype(ref, key);   // the file dtype is converted from even when `dtype` is set
                nb::gil_scoped_release rel;      // disk read + convert, pure C++
                return vai::Tensor(ref, dtype.value_or(vai::Dtype::None));
            },
            nb::arg("key"),
            nb::arg("dtype") = nb::none(),
            "Load `key` as a Tensor, optionally converting to `dtype` at load "
            "time. Pass `dtype=None` (default) to keep the file's native dtype."
        )

        .def(
            "ref",
            [](const vai::SafetensorsParser& self, const std::string& key) -> vai::SafetensorRef {
                auto ref = self[key];
                ensureKeyExists(ref, key);
                return ref;
            },
            nb::arg("key"),
            nb::keep_alive<0, 1>(),   // the ref keeps this parser alive
            "Metadata handle for `key`: .shape and .dtype without loading the "
            "tensor data. Raises KeyError for an unknown key."
        )

        .def(
            "__repr__",
            [](const vai::SafetensorsParser& self) -> std::string {
                auto names = self.getTensorNames();
                return "SafetensorsParser(num_tensors=" +
                       std::to_string(names.size()) + ")";
            }
        );
}

}   // namespace pyvai