#include "pyvai-internal.h"
#include "vai/engine/graph-flow.h"   // NodeFlow / NodeGroup — split out of neural-net.h
#include "vai/engine/neural-net.h"

#include <stdexcept>
#include <vector>

namespace nb = nanobind;

namespace pyvai {

struct NodeGroupAccess : vai::NodeGroup
{
    using vai::NodeGroup::defineSlot;
};


// DSL lifetime model — the core keeps raw NodeSlot* both ways and assumes
// nodes outlive the graph, so the Python rule is "a node wired into a net
// lives as long as the net": net.input()/output() wrappers carry a
// `_pyvai_net` anchor (a strong ref), flows drag strong refs to every
// wrapper wired through them, and each connection adopts the chain into
// `net._pyvai_dsl_refs` once either side is anchored — temporaries like
// `net.input() - nodes.IdentityNode() - net.output()` survive the statement.
// Slot lookups funnel here so a wrong name reports the node's real slots
// instead of the core map's bare out_of_range ("map::at").
[[noreturn]] inline void throwNoSuchSlot(const vai::Node& n, const std::string& name)
{
    std::string msg = "no such slot '" + name + "' (slots:";
    const char* sep = " ";
    for (const auto& [slotName, slot] : n.slots())
    {
        msg += sep;
        msg += slotName;
        sep = ", ";
    }
    msg += ")";
    throw nb::value_error(msg.c_str());
}


struct PyNodeFlow
{
    std::string     inSlotName  = "in0";
    vai::Node*      node        = nullptr;
    vai::NodeGroup* group       = nullptr;
    std::string     outSlotName = "out0";

    // Strong refs to every wrapper wired through this chain (endpoint included).
    std::vector<nb::object> refs;

    vai::NodeSlot& slotByName(const std::string& name) const
    {
        if (node)
        {
            if (!node->findSlot(name))
                throwNoSuchSlot(*node, name);
            return node->slot(name);
        }
        try
        {
            return group->slot(name);
        }
        catch (const std::out_of_range&)
        {
            throw nb::value_error(
                ("no such slot '" + name + "' on this group").c_str());
        }
    }
};


namespace {

PyNodeFlow flowOf(nb::handle_t<vai::Node> h)
{
    PyNodeFlow f;
    f.node = &nb::cast<vai::Node&>(h);
    f.refs = { nb::borrow<nb::object>(h) };
    return f;
}

PyNodeFlow flowOf(nb::handle_t<vai::NodeGroup> h)
{
    PyNodeFlow f;
    f.group = &nb::cast<vai::NodeGroup&>(h);
    f.refs  = { nb::borrow<nb::object>(h) };
    return f;
}

// The net anchoring this chain, or a null object when it has none yet.
nb::object anchorNetOf(const PyNodeFlow& f)
{
    for (const nb::object& o : f.refs)
        if (nb::hasattr(o, "_pyvai_net"))
            return o.attr("_pyvai_net");
    return nb::object();
}

// Stamp + stash every not-yet-anchored wrapper of `f` so the net owns it.
void adoptInto(nb::object net, const PyNodeFlow& f)
{
    nb::list bag;
    if (nb::hasattr(net, "_pyvai_dsl_refs"))
        bag = nb::cast<nb::list>(net.attr("_pyvai_dsl_refs"));
    else
        net.attr("_pyvai_dsl_refs") = bag;

    for (const nb::object& o : f.refs)
        if (!nb::hasattr(o, "_pyvai_net"))
        {
            o.attr("_pyvai_net") = net;
            bag.append(o);
        }
}

PyNodeFlow connectFlowToFlow(const PyNodeFlow& src, const PyNodeFlow& dst)
{
    vai::NodeSlot& srcSlot = src.slotByName(src.outSlotName);
    vai::NodeSlot& dstSlot = dst.slotByName(dst.inSlotName);
    // The core connect() asserts (aborts) on slot-type mismatch.
    if (srcSlot.type() != vai::NodeSlot::output || dstSlot.type() != vai::NodeSlot::input)
        throw nb::value_error(
            ("connect: '" + src.outSlotName + "' must be an output slot and '" +
             dst.inSlotName + "' an input slot").c_str());
    vai::NodeSlot::connect(srcSlot, dstSlot);

    nb::object net = anchorNetOf(src);
    if (!net.is_valid()) net = anchorNetOf(dst);
    if (net.is_valid())
    {
        adoptInto(net, src);
        adoptInto(net, dst);
    }

    // Chaining continues from dst; carry the whole chain's refs so a later
    // connection that reaches a net can adopt everything at once.
    PyNodeFlow out = dst;
    out.refs.insert(out.refs.end(), src.refs.begin(), src.refs.end());
    return out;
}

// Any wirable object (Node / NodeGroup / NodeFlow) as a flow.
PyNodeFlow toFlow(nb::handle h)
{
    if (nb::isinstance<PyNodeFlow>(h))
        return nb::cast<const PyNodeFlow&>(h);
    PyNodeFlow f;
    if (nb::isinstance<vai::Node>(h))
        f.node = &nb::cast<vai::Node&>(h);
    else if (nb::isinstance<vai::NodeGroup>(h))
        f.group = &nb::cast<vai::NodeGroup&>(h);
    else
        throw nb::type_error("connect: expected a Node, NodeGroup or NodeFlow");
    f.refs = { nb::borrow<nb::object>(h) };
    return f;
}

// Call wiring: `self(x, ...)` connects each arg into `self` and
// returns self's flow — sugar over `-`. Positional args go to in0, in1, ...;
// keyword args go to the named slots (e.g. attn(Q=..., K=..., V=...)).
PyNodeFlow connectCall(PyNodeFlow out, nb::args args, nb::kwargs kwargs)
{
    if (nb::len(args) == 0 && nb::len(kwargs) == 0)
        throw nb::type_error("node(...): pass at least one upstream node or flow");
    size_t i = 0;
    for (nb::handle a : args)
    {
        PyNodeFlow dst = out;
        dst.inSlotName = "in" + std::to_string(i++);
        out = connectFlowToFlow(toFlow(a), dst);
    }
    for (auto [k, v] : kwargs)
    {
        PyNodeFlow dst = out;
        dst.inSlotName = nb::cast<std::string>(k);
        out = connectFlowToFlow(toFlow(v), dst);
    }
    return out;
}

// PyNodeFlow holds nb::objects in a C++ vector the cyclic GC cannot see.
// Chains like `self.flow = self.input(..) - node` form cycles through those
// refs (flow -> node wrapper -> _pyvai_net -> net -> __dict__ -> flow), so
// NodeFlow implements tp_traverse/tp_clear to make them collectable.
int nodeFlowTraverse(PyObject* self, visitproc visit, void* arg)
{
    if (!nb::inst_ready(self))
        return 0;
    PyNodeFlow* f = nb::inst_ptr<PyNodeFlow>(self);
    for (const nb::object& o : f->refs)
        Py_VISIT(o.ptr());
    Py_VISIT(Py_TYPE(self));
    return 0;
}

int nodeFlowClear(PyObject* self)
{
    if (nb::inst_ready(self))
        nb::inst_ptr<PyNodeFlow>(self)->refs.clear();
    return 0;
}

PyType_Slot nodeFlowSlots[] = {
    { Py_tp_traverse, reinterpret_cast<void*>(nodeFlowTraverse) },
    { Py_tp_clear,    reinterpret_cast<void*>(nodeFlowClear) },
    { 0, nullptr }
};

void checkSlotExists(vai::Node& n, const std::string& name)
{
    if (!n.findSlot(name))
        throwNoSuchSlot(n, name);
}

}   // namespace


void bindBaseNode(nb::module_& m)
{
    // NodeSlot — opaque handle. No methods needed; consumed only by
    // NodeGroup.define_slot and the connection operators.
    nb::class_<vai::NodeSlot>(m, "NodeSlot");

    // dynamic_attr: carries the `_pyvai_net` anchor mark (see PyNodeFlow).
    nb::class_<vai::Node>(m, "Node", nb::dynamic_attr())
        .def_prop_rw(
            "profile_tag",
            [](const vai::Node& self) -> const std::string& { return self.getProfileTag(); },
            [](vai::Node& self, const std::string& tag) { self.setProfileTag(tag); },
            "Profile tag for this node; tagging any node turns per-node GPU "
            "timing on for its net. A tag may be set or changed at any point "
            "(the net re-records); an empty tag untags the node. Changing the "
            "tag set drops the running averages, which are then incomparable."
        )

        .def(
            "slot",
            [](vai::Node& self, const std::string& name) -> vai::NodeSlot& {
                checkSlotExists(self, name);
                return self.slot(name);
            },
            nb::arg("name"),
            nb::rv_policy::reference_internal
        )

        .def(
            "__getitem__",
            [](vai::Node& self, const std::string& name) -> vai::Tensor {
                checkSlotExists(self, name);
                return self[name];
            }
        )

        .def(
            "__setitem__",
            [](vai::Node& self, const std::string& name, const vai::Tensor& t) {
                checkSlotExists(self, name);
                self[name] = t;
            }
        )

        // a - b  →  connect a.out0 to b.in0; return b's flow for chaining.
        .def(
            "__sub__",
            [](nb::handle_t<vai::Node> a, nb::handle_t<vai::Node> b) -> PyNodeFlow {
                return connectFlowToFlow(flowOf(a), flowOf(b));
            }
        )
        .def(
            "__sub__",
            [](nb::handle_t<vai::Node> a, nb::handle_t<vai::NodeGroup> b) -> PyNodeFlow {
                return connectFlowToFlow(flowOf(a), flowOf(b));
            }
        )
        .def(
            "__sub__",
            [](nb::handle_t<vai::Node> a, const PyNodeFlow& b) -> PyNodeFlow {
                return connectFlowToFlow(flowOf(a), b);
            }
        )

        // Call-style alternative to `-`: y = linear(relu(x))
        .def(
            "__call__",
            [](nb::handle_t<vai::Node> self, nb::args args, nb::kwargs kwargs) -> PyNodeFlow {
                return connectCall(flowOf(self), args, kwargs);
            },
            "Wire the given nodes/flows into this node and return its flow — "
            "call-syntax sugar over `-`. Positional args connect to in0, "
            "in1, ...; keyword args connect to the named slots."
        )

        // node / "name"   →  out-slot spec
        .def(
            "__truediv__",
            [](nb::handle_t<vai::Node> self, const std::string& name) -> PyNodeFlow {
                PyNodeFlow f = flowOf(self);
                f.outSlotName = name;
                return f;
            }
        )
        // "name" / node   →  in-slot spec
        .def(
            "__rtruediv__",
            [](nb::handle_t<vai::Node> self, const std::string& name) -> PyNodeFlow {
                PyNodeFlow f = flowOf(self);
                f.inSlotName = name;
                return f;
            }
        )

        .def(
            "__repr__",
            [](nb::handle self) -> std::string {
                auto& n = nb::cast<vai::Node&>(self);
                std::string typeName = nb::cast<std::string>(
                    self.type().attr("__name__"));
                const auto& tag = n.getProfileTag();
                if (tag.empty())
                    return typeName + "()";
                return typeName + "(profile_tag='" + tag + "')";
            }
        );

    nb::class_<vai::InputNode,  vai::Node>(m, "InputNode");
    nb::class_<vai::OutputNode, vai::Node>(m, "OutputNode");

    nb::class_<vai::NodeGroup>(m, "NodeGroup", nb::dynamic_attr())
        .def(nb::init<>())

        .def(
            "slot",
            [](vai::NodeGroup& self, const std::string& name) -> vai::NodeSlot& {
                return self.slot(name);
            },
            nb::arg("name"),
            nb::rv_policy::reference_internal
        )

        // Member-function-pointer trick to expose protected NodeGroup::defineSlot.
        // keep_alive<1,3>: self holds the slot wrapper (arg 3 — NOT 2, the
        // name), which holds its node via slot()'s reference_internal — so a
        // group built from local nodes keeps them alive.
        .def("define_slot", &NodeGroupAccess::defineSlot,
             nb::arg("name"), nb::arg("slot"), nb::keep_alive<1, 3>())

        .def(
            "__sub__",
            [](nb::handle_t<vai::NodeGroup> a, nb::handle_t<vai::Node> b) -> PyNodeFlow {
                return connectFlowToFlow(flowOf(a), flowOf(b));
            }
        )
        .def(
            "__sub__",
            [](nb::handle_t<vai::NodeGroup> a, nb::handle_t<vai::NodeGroup> b) -> PyNodeFlow {
                return connectFlowToFlow(flowOf(a), flowOf(b));
            }
        )
        .def(
            "__sub__",
            [](nb::handle_t<vai::NodeGroup> a, const PyNodeFlow& b) -> PyNodeFlow {
                return connectFlowToFlow(flowOf(a), b);
            }
        )

        .def(
            "__call__",
            [](nb::handle_t<vai::NodeGroup> self, nb::args args, nb::kwargs kwargs) -> PyNodeFlow {
                return connectCall(flowOf(self), args, kwargs);
            },
            "Wire the given nodes/flows into this group and return its flow — "
            "call-syntax sugar over `-`. Positional args connect to in0, "
            "in1, ...; keyword args connect to the named slots."
        )

        .def(
            "__truediv__",
            [](nb::handle_t<vai::NodeGroup> self, const std::string& name) -> PyNodeFlow {
                PyNodeFlow f = flowOf(self);
                f.outSlotName = name;
                return f;
            }
        )
        .def(
            "__rtruediv__",
            [](nb::handle_t<vai::NodeGroup> self, const std::string& name) -> PyNodeFlow {
                PyNodeFlow f = flowOf(self);
                f.inSlotName = name;
                return f;
            }
        )

        .def(
            "__repr__",
            [](nb::handle self) -> std::string {
                std::string typeName = nb::cast<std::string>(
                    self.type().attr("__name__"));
                return typeName + "()";
            }
        );

    nb::class_<PyNodeFlow>(m, "NodeFlow", nb::type_slots(nodeFlowSlots))
        .def(
            "__sub__",
            [](const PyNodeFlow& a, nb::handle_t<vai::Node> b) -> PyNodeFlow {
                return connectFlowToFlow(a, flowOf(b));
            }
        )
        .def(
            "__sub__",
            [](const PyNodeFlow& a, nb::handle_t<vai::NodeGroup> b) -> PyNodeFlow {
                return connectFlowToFlow(a, flowOf(b));
            }
        )
        .def(
            "__sub__",
            [](const PyNodeFlow& a, const PyNodeFlow& b) -> PyNodeFlow {
                return connectFlowToFlow(a, b);
            }
        )
        .def(
            "__truediv__",
            [](const PyNodeFlow& self, const std::string& name) -> PyNodeFlow {
                PyNodeFlow out = self;
                out.outSlotName = name;
                return out;
            }
        )
        // "name" / flow   →  in-slot spec, same as on Node / NodeGroup
        .def(
            "__rtruediv__",
            [](const PyNodeFlow& self, const std::string& name) -> PyNodeFlow {
                PyNodeFlow out = self;
                out.inSlotName = name;
                return out;
            }
        )

        .def(
            "__repr__",
            [](const PyNodeFlow& f) -> std::string {
                std::string target;
                if (f.node)
                {
                    const auto& tag = f.node->getProfileTag();
                    target = "node=<Node" +
                             (tag.empty() ? std::string() : " '" + tag + "'") +
                             ">";
                }
                else if (f.group)
                    target = "group=<NodeGroup>";
                return "NodeFlow(" + target +
                       ", in='" + f.inSlotName +
                       "', out='" + f.outSlotName + "')";
            }
        );
}

}   // namespace pyvai
