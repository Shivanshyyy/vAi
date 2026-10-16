#include "bind-kvcache-counter.h"

#include <cstdint>
#include <optional>
#include <string>

namespace pyvai {

void bindKVCacheCounter(nb::module_& m)
{
    // The buffer must live on the same device as the net whose nodes consume
    // it — see the ctor docstring.
    nb::class_<KVCacheCounter>(m, "KVCacheCounter")
        .def(
            "__init__",
            [](KVCacheCounter* self, std::optional<eva::Device> device) {
                eva::Device dev = device.value_or(eva::Device{});
                if (!dev)
                    dev = vai::getVaiDevice();
                new (self) KVCacheCounter(
                    dev.createBuffer({
                        .size       = sizeof(uint32_t),
                        .usage      = eva::BUFFER_USAGE::UNIFORM_BUFFER,
                        .reqMemProps = eva::MEMORY_PROPERTY::HOST_VISIBLE
                                     | eva::MEMORY_PROPERTY::HOST_COHERENT,
                    }));
                *reinterpret_cast<uint32_t*>(self->buf.map()) = 0;
            },
            nb::arg("device") = nb::none(),
            "Create the counter's uint32 buffer on `device` (default: the "
            "process-wide get_vai_device()). Use device=net.device when the "
            "consuming net was built with its own device."
        )
        .def("update", [](KVCacheCounter& self, uint32_t count) {
            *reinterpret_cast<uint32_t*>(self.buf.map()) = count;
        }, nb::arg("count"),
            "Set the KV-cache count. Not synchronized — call "
            "net.sync_prev_forward() first if a forward may still be reading "
            "the counter (the C++ models do the same).")
        .def("reset", [](KVCacheCounter& self) {
            *reinterpret_cast<uint32_t*>(self.buf.map()) = 0;
        }, "Zero the count. Same synchronization rule as update().")
        .def_prop_ro("value", [](KVCacheCounter& self) -> uint32_t {
            return *reinterpret_cast<uint32_t*>(self.buf.map());
        })
        .def("__repr__", [](KVCacheCounter& self) -> std::string {
            uint32_t v = *reinterpret_cast<uint32_t*>(self.buf.map());
            return "KVCacheCounter(value=" + std::to_string(v) + ")";
        });
}

}   // namespace pyvai
