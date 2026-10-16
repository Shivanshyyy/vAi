#include "pyvai-internal.h"

#include <nanobind/nanobind.h>


namespace nb = nanobind;


NB_MODULE(_pyvai, m)
{
    m.doc() = "vAi engine — Python bindings.";

    pyvai::bindEngine(m);
    pyvai::bindTensor(m);
    pyvai::bindBaseNode(m);
    pyvai::bindNodes(m);
    pyvai::bindNeuralNet(m);
    pyvai::bindSafetensors(m);
    pyvai::bindModels(m);
}