#include "pyvai-internal.h"

#include "vai/engine/neural-net.h"

#include <string>
#include <vector>


namespace nb = nanobind;

namespace pyvai {

void bindNeuralNet(nb::module_& m)
{
    // NOTE: the core's shared BufferPool/PipelinePool are not exposed — the
    // experimental bindings assume one model, one NeuralNet per process.
    // KNOWN LIMIT: __call__ with wrong input-dict keys hits the core's
    // VAI_ASSERT (debug builds abort the interpreter) — unguardable here
    // until the core exposes the net's input names.
    nb::class_<vai::NeuralNet>(m, "NeuralNet", nb::dynamic_attr())
        .def(
            "__init__",
            [](vai::NeuralNet* self,
               vai::Dtype w_default_dtype,
               vai::Dtype o_default_dtype,
               size_t upload_chunk_size_mib,
               uint32_t buffer_pool_log_level,
               std::optional<eva::Device> device)
            {
                vai::NeuralNetConfig config;
                config.device             = device.value_or(eva::Device{});
                config.wDefaultDtype      = w_default_dtype;
                config.oDefaultDtype      = o_default_dtype;
                config.uploadChunkSizeMiB = upload_chunk_size_mib;
                config.bufferPoolLogLevel = buffer_pool_log_level;
                new (self) vai::NeuralNet(config);
            },
            nb::arg("w_default_dtype")       = vai::Dtype::Fp16,
            nb::arg("o_default_dtype")       = vai::Dtype::Fp32,
            nb::arg("upload_chunk_size_mib") = static_cast<size_t>(256),
            nb::arg("buffer_pool_log_level") = 0u,
            nb::arg("device")                = nb::none(),
            "Build an empty net. `w_default_dtype` is the fallback for layer "
            "parameters (weights, KV cache); `o_default_dtype` is the fallback "
            "for layer outputs. A layer may override either. `device` "
            "is where the net's buffers, pipelines and command buffer live "
            "(default: the process-wide get_vai_device()). "
            "`buffer_pool_log_level` applies to the net's internal buffer pool."
        )

        .def_prop_ro(
            "w_default_dtype",
            [](const vai::NeuralNet& self) { return self.config().wDefaultDtype; }
        )
        .def_prop_ro(
            "o_default_dtype",
            [](const vai::NeuralNet& self) { return self.config().oDefaultDtype; }
        )
        .def_prop_ro(
            "device",
            [](const vai::NeuralNet& self) { return self.device(); },
            "The Device this net lives on. Pass it to whatever must share the "
            "net's device — e.g. KVCacheCounter(device=net.device)."
        )

        // input()/output() wrappers carry `_pyvai_net`: a strong ref back to
        // the net (replacing reference_internal) that also marks them as
        // net-anchored for the DSL lifetime model — connecting anything to an
        // anchored wrapper makes the net own it (see bind-base-node.cpp).
        .def(
            "input",
            [](nb::handle_t<vai::NeuralNet> self, const std::string& name) -> nb::object {
                auto& net = nb::cast<vai::NeuralNet&>(self);
                nb::object o = nb::cast(net.input(name), nb::rv_policy::reference);
                o.attr("_pyvai_net") = self;
                return o;
            },
            nb::arg("name") = "in0",
            nb::sig("def input(self, name: str = 'in0') -> InputNode"),
            "Get (or lazily create) the named input boundary node, e.g. "
            "net.input(\"in0\"). Names are the keys of __call__'s input dict."
        )
        .def(
            "output",
            [](nb::handle_t<vai::NeuralNet> self, const std::string& name) -> nb::object {
                auto& net = nb::cast<vai::NeuralNet&>(self);
                nb::object o = nb::cast(net.output(name), nb::rv_policy::reference);
                o.attr("_pyvai_net") = self;
                return o;
            },
            nb::arg("name") = "out0",
            nb::sig("def output(self, name: str = 'out0') -> OutputNode"),
            "Get (or lazily create) the named output boundary node, e.g. "
            "net.output(\"out0\"). Names are the keys of __call__'s result dict."
        )

        .def(
            "invalidate_command_buffer",
            &vai::NeuralNet::invalidateCommandBuffer
        )

        .def(
            "sync_prev_forward",
            [](vai::NeuralNet& self) {
                nb::gil_scoped_release rel;
                self.syncPrevForward();
            },
            "Block until the in-flight forward (if any) completes. Safe when idle."
        )

        // Profiling turns itself on: tag a node (node.profile_tag = "...") and
        // the net starts timing it. With nothing tagged these are no-ops
        // rather than errors.
        .def(
            "reset_profile_records",
            [](vai::NeuralNet& self) {
                nb::gil_scoped_release rel;   // waits the in-flight forward
                self.resetProfileRecords();
            },
            "Clear the running records (e.g. after warm-up). Tags stay in "
            "place — clear_profile_tags() is what turns profiling off."
        )
        .def(
            "print_profile",
            [](vai::NeuralNet& self, bool reset_records) {
                nb::gil_scoped_release rel;
                self.printProfile(reset_records);
            },
            nb::arg("reset_records") = true,
            "Dump the per-node timing tree to stdout, clearing the records "
            "afterwards by default. Prints nothing when no node is tagged."
        )
        .def(
            "gpu_time_ms",
            [](vai::NeuralNet& self, const std::string& tag) -> double {
                nb::gil_scoped_release rel;
                return self.getGpuTimeMs(tag);
            },
            nb::arg("tag"),
            "Average elapsed ms for one profile tag, over the forwards since "
            "the records were last cleared. 0.0 for an unknown or untagged tag."
        )
        .def(
            "clear_profile_tags",
            &vai::NeuralNet::clearProfileTags,
            "Untag every node, turning profiling off. Retagging afterwards "
            "starts from a clean set."
        )

        .def(
            "__call__",
            [](nb::handle_t<vai::NeuralNet> self, nb::dict inputs) -> nb::dict {
                auto& net = nb::cast<vai::NeuralNet&>(self);

                vai::TensorDict in;
                for (auto [k, v] : inputs)
                {
                    if (!nb::isinstance<nb::str>(k) || !nb::isinstance<vai::Tensor>(v))
                        throw nb::type_error("__call__: inputs must map str -> Tensor");
                    in.emplace(nb::cast<std::string>(k), nb::cast<const vai::Tensor&>(v));
                }

                vai::TensorDict out;
                {
                    nb::gil_scoped_release rel;
                    out = net(std::move(in));
                }

                // Guard bookkeeping — only after a successful forward.
                uint64_t gen = 1;
                if (nb::hasattr(self, "_pyvai_forward_gen"))
                    gen = nb::cast<uint64_t>(self.attr("_pyvai_forward_gen")) + 1;
                self.attr("_pyvai_forward_gen") = gen;

                nb::dict res;
                for (auto& [name, t] : out)
                {
                    nb::object o = nb::cast(t);
                    o.attr("_pyvai_net")     = self;     // also keeps the net alive under its outputs
                    o.attr("_pyvai_out_gen") = gen;
                    o.attr("_pyvai_read")    = false;
                    res[name.c_str()] = o;
                }
                return res;
            },
            nb::arg("inputs")
        )

        .def(
            "__repr__",
            [](nb::handle self) -> std::string {
                std::string typeName = nb::cast<std::string>(
                    self.type().attr("__name__"));
                return typeName + "()";
            }
        );
}

}   // namespace pyvai