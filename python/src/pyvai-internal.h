#ifndef PYVAI_INTERNAL_H
#define PYVAI_INTERNAL_H

#include "binding-core.h"

#include <filesystem>
#include <string>


namespace pyvai {

inline void ensureFileExists(const char* who, const std::string& path)
{
    std::error_code ec;
    if (!std::filesystem::exists(path, ec) || ec)
    {
        PyErr_SetString(PyExc_FileNotFoundError,
            (std::string(who) + ": file not found: " + path).c_str());
        throw nanobind::python_error();
    }
}

void bindEngine(nanobind::module_&);
void bindTensor(nanobind::module_&);
void bindBaseNode(nanobind::module_&);
void bindNodes(nanobind::module_&);
void bindNeuralNet(nanobind::module_&);
void bindSafetensors(nanobind::module_&);
void bindModels(nanobind::module_&);

}   // namespace pyvai

#endif  // PYVAI_INTERNAL_H