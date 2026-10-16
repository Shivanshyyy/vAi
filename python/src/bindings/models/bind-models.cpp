#include "pyvai-internal.h"

#include "bind-models.h"


namespace pyvai {

void bindModels(nb::module_& m)
{
    // GPT-2 and Llama 3 both take this as an argument type, so register it first.
    bindTokenizer(m);

    bindMnist(m);
    bindGPT2(m);
    bindLlama3(m);
    bindYOLOv8(m);
}

}   // namespace pyvai