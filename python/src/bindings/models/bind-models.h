#ifndef PYVAI_BIND_MODELS_H
#define PYVAI_BIND_MODELS_H

#include "binding-core.h"

#include "vai/engine/safetensors-parser.h"



namespace pyvai {

void bindTokenizer(nb::module_&);
void bindMnist(nb::module_&);
void bindGPT2(nb::module_&);
void bindLlama3(nb::module_&);
void bindYOLOv8(nb::module_&);

}   // namespace pyvai

#endif  // PYVAI_BIND_MODELS_H