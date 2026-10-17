#include "python/bindings/module.h"
#include "python/bindings/models/models.h"


namespace pyvai {

void bindModels(nb::module_& m)
{
    bindModelConfigs(m);   // the families' configs are built from these
    bindGPT2(m);
    bindLlama3(m);
    bindGemma3(m);
    bindPaliGemma(m);
    bindPi0(m);
    bindPi05(m);
    bindYOLOX(m);
}

}   // namespace pyvai
