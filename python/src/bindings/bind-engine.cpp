#include "pyvai-internal.h"

#ifdef __GLIBC__
#include <stdio_ext.h>   // __fpurge
#include <unistd.h>
#elif defined(_WIN32)
#include <io.h>          // _isatty, _fileno
#endif

#include "vai/engine/neural-net.h"

#include <optional>
#include <string>

// Defined in neural-net.cpp with external linkage; no header declares it.
namespace vai { extern eva::DescriptorPool gVaiDestSetPool; }


namespace pyvai {

namespace {

// On Windows, ExitProcess kills the driver's worker threads BEFORE the DLL
// detach that runs eva's static ~Runtime — its vkDestroyDevice then crashes.
// Destroying the device at Py_AtExit (after Python objects, before
// ExitProcess) leaves ~Runtime a null-guarded no-op. Harmless on Linux;
// bypassed by os._exit.
void shutdownEngineAtExit()
{
    if (!vai::gVaiDestSetPool)        // init probe — getVaiDevice() would lazy-init
        return;
    vai::gVaiDestSetPool.destroy();   // device child — destroy first
    vai::getVaiDevice().destroy();    // idempotent; waits the GPU idle inside
}

}   // namespace

void bindEngine(nb::module_& m)
{
    Py_AtExit(shutdownEngineAtExit);

    // eva::Device — opaque handle to the Vulkan device a net (with its buffers,
    // pipelines and command buffer) lives on. Python can only obtain one from
    // get_vai_device() or NeuralNet.device and hand it back to the device=
    // parameters; there is deliberately no Python-side physical-GPU picker —
    // eva has none (Runtime::createDevice prompts on stdin when several GPUs
    // are present).
    nb::class_<eva::Device>(m, "Device")
        .def("__bool__",
             [](const eva::Device& self) { return bool(self); },
             "True when this handle is bound to a live device.")
        .def("__eq__",
             [](const eva::Device& self, nb::handle other) -> bool {
                 if (!nb::isinstance<eva::Device>(other))
                     return false;
                 return self == nb::cast<eva::Device>(other);
             },
             nb::arg("other"))
        .def("__repr__", [](const eva::Device& self) -> std::string {
            return self ? "Device(<bound>)" : "Device(<null>)";
        });

    m.def("init_vai_engine",
        [](std::optional<eva::Device> device) {
            vai::initVAIEngine(device.value_or(eva::Device{}));
            // eva's device prompt (scanf) leaves its newline in C stdio, and
            // Python's input() on a tty reads through the same FILE — the
            // first prompt would fire empty. A tty buffers at most that one
            // line, so purging here is safe; never blocks.
#ifdef __GLIBC__
            if (isatty(fileno(stdin)))
                __fpurge(stdin);
#elif defined(_WIN32)
            // ucrt documents fflush on an input stream as clearing its buffer.
            if (_isatty(_fileno(stdin)))
                fflush(stdin);
#endif
        },
        nb::arg("device") = nb::none(),
        "Initialize the vAi engine. Idempotent — on repeat calls `device` is "
        "ignored. Pass a Device to bind the engine to it; by default eva "
        "picks one (prompting on stdin when several GPUs are present).");

    m.def("get_vai_device",
        []() { return vai::getVaiDevice(); },
        "The process-wide default Device (initializing the engine on first "
        "use). Nets built without device= live on this one.");
}

}  // namespace pyvai
