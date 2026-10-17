#ifndef PYVAI_COMMON_H
#define PYVAI_COMMON_H

// What every binding file starts from: nanobind and its casters. One naming
// rule is the bindings' own: the lambda behind a .def names its parameters
// after the Python keywords they carry (nb::arg("kernel_size") <-> kernel_size),
// so those read in snake_case. Everything else is lowerCamel, as in the core.
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/array.h>
#include <nanobind/stl/map.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/shared_ptr.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

namespace nb = nanobind;

#endif // PYVAI_COMMON_H
