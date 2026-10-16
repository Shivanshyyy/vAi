#include "bind-nodes.h"

#include "vai/nodes/permute.h"

#include <cstdint>
#include <optional>
#include <vector>

namespace pyvai {

void bindPermute(nb::module_& m)
{
    nb::class_<vai::node::Permute, vai::Node>(m, "PermuteNode")
        .def(
            "__init__",
            [](vai::node::Permute* self,
               std::vector<uint32_t> perms,
               std::optional<vai::Dtype> o_dtype)
            {
                requireCurrentNet("PermuteNode");
                // The core stores perms unvalidated and indexes with them —
                // out-of-range or duplicate entries are an OOB read at prepare.
                const size_t rank = perms.size();
                if (rank == 0 || rank > PERMUTE_MAX_RANK)
                    throw nb::value_error(
                        "PermuteNode: perms must hold between 1 and 8 axes");
                std::vector<bool> seen(rank, false);
                for (uint32_t axis : perms)
                {
                    if (axis >= rank || seen[axis])
                        throw nb::value_error(
                            "PermuteNode: perms must be a permutation of (0, .., len(perms) - 1)");
                    seen[axis] = true;
                }
                new (self) vai::node::Permute(
                    std::move(perms),
                    vai::node::Permute::Config{
                        .oDtype = o_dtype.value_or(vai::Dtype::None),
                    });
            },
            nb::arg("perms"),
            nb::arg("o_dtype") = nb::none(),
            "perms is a tuple of distinct axis indices, e.g. (1, 0, 2): out0 "
            "dimension i is in0 dimension perms[i], and its length fixes the rank "
            "in0 must have. out0 aliases in0's buffer when the permutation preserves "
            "memory order and the dtype is unchanged, and takes new contiguous "
            "storage otherwise."
        );
}

}   // namespace pyvai
