#include "pyvai-internal.h"

#include "bind-models.h"

#include "vai/utils/tokenizer.h"   // vai/utils — one tokenizer for every model

#include <string>
#include <vector>


namespace pyvai {

void bindTokenizer(nb::module_& m)
{
    nb::class_<Tokenizer>(m, "Tokenizer")
        .def(
            "__init__",
            [](Tokenizer* self, const std::string& path) {
                ensureFileExists("Tokenizer", path);
                new (self) Tokenizer(path);
            },
            nb::arg("tokenizer_json"),
            "Load a HuggingFace tokenizer.json (the same file the C++ programs read)."
        )

        // BPE is pure C++ after argument conversion — drop the GIL.
        .def("encode", &Tokenizer::encode,
             nb::arg("text"),
             nb::arg("add_special")   = false,
             nb::arg("parse_special") = true,
             nb::call_guard<nb::gil_scoped_release>())
        .def("decode", &Tokenizer::decode,
             nb::arg("ids"),
             nb::arg("skip_special") = true,
             nb::call_guard<nb::gil_scoped_release>())

        .def("vocab_size", &Tokenizer::vocabSize)

        // Which id ends a turn is a property of the checkpoint, not the
        // tokenizer: read the names out of generation_config.json and resolve
        // them here, e.g. piece_to_id("<|eot_id|>").
        .def("piece_to_id", &Tokenizer::pieceToId, nb::arg("piece"),
             "Returns -1 for an unknown piece.")
        .def("id_to_piece", &Tokenizer::idToPiece, nb::arg("id"),
             "Returns '' for an unknown id.")
        .def("is_special",  &Tokenizer::isSpecial, nb::arg("id"))

        .def("__repr__", [](const Tokenizer& self) -> std::string {
            return "Tokenizer(vocab_size=" + std::to_string(self.vocabSize()) + ")";
        });
}

}   // namespace pyvai
