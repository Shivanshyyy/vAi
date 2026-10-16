#include "pyvai-internal.h"

#include "bind-models.h"
#include "utils/yolov8.h"
#include "utils/yolov8-processing.h"   // Detection + pre/postprocess (temporary copy)

#include "models/yolov8/model.h"
#include "models/yolov8/ultralytics-weights.h"


namespace pyvai {

void bindYOLOv8(nb::module_& m)
{
    nb::enum_<yolov8::Tier>(m, "YOLOv8Tier")
        .value("n", yolov8::Tier::NANO)
        .value("s", yolov8::Tier::SMALL)
        .value("m", yolov8::Tier::MEDIUM)
        .value("l", yolov8::Tier::LARGE)
        .value("x", yolov8::Tier::XLARGE);

    nb::class_<Detection>(m, "Detection")
        .def_ro("x1", &Detection::x1)
        .def_ro("y1", &Detection::y1)
        .def_ro("x2", &Detection::x2)
        .def_ro("y2", &Detection::y2)
        .def_ro("confidence", &Detection::confidence)
        .def_ro("class_id", &Detection::classId)
        .def_ro("class_name", &Detection::className)
        .def("__repr__", [](const Detection& d) {
            return "<Detection " + d.className + " "
                 + std::to_string(d.confidence) + ">";
        });

    nb::class_<yolov8::Model, vai::NeuralNet>(m, "YOLOv8Model")
        .def(
            "__init__",
            [](yolov8::Model* self,
               yolov8::Tier tier,
               uint32_t image_size, float conf_threshold, float iou_threshold,
               vai::Dtype w_default_dtype,
               vai::Dtype o_default_dtype,
               std::optional<eva::Device> device)
            {
                // Concrete dtypes and a legal image size, or the model's node
                // ctors / first forward assert (aborting the interpreter).
                if (w_default_dtype == vai::Dtype::None || o_default_dtype == vai::Dtype::None)
                    throw nb::value_error("YOLOv8Model: dtypes must be F32 or F16, not NONE");
                if (image_size % 32 != 0 || image_size < 32 || image_size > 1024)
                    throw nb::value_error(
                        "YOLOv8Model: image_size must be a multiple of 32 in [32, 1024]");
                yolov8::ModelConfig mcfg = yolov8::getConfigCOCO(tier);
                yolov8::RuntimeConfig run{
                    .imageSize     = image_size,
                    .confThreshold = conf_threshold,
                    .iouThreshold  = iou_threshold,
                };
                vai::NeuralNetConfig cfg;
                cfg.device        = device.value_or(eva::Device{});
                cfg.wDefaultDtype = w_default_dtype;
                cfg.oDefaultDtype = o_default_dtype;
                new (self) yolov8::Model(cfg, mcfg, run);
            },
            nb::arg("tier")            = yolov8::Tier::LARGE,
            nb::arg("image_size")      = 640u,
            nb::arg("conf_threshold")  = 0.25f,
            nb::arg("iou_threshold")   = 0.45f,
            nb::arg("w_default_dtype") = vai::Dtype::Fp16,
            nb::arg("o_default_dtype") = vai::Dtype::Fp32,
            nb::arg("device")          = nb::none(),
            "Build a YOLOv8 model (COCO config). image_size must be a "
            "multiple of 32, at most 1024."
        )
        .def(
            "load_weight",
            [](yolov8::Model& self, std::optional<std::string> path)
            {
                // Default: the path loadWeights<"ultralytics"> composes
                // (models/yolov8/ultralytics-weights.h) — data/yolov8/<tier>/yolov8.safetensors.
                std::string p = path.value_or(
                    std::string(VAI_DATA_DIR "/yolov8/")
                    + "nsmlx"[(int)self.cfg.tier] + "/yolov8.safetensors");
                ensureFileExists("load_weight", p);
                // Pure C++ from here — parse + upload can take seconds.
                nb::gil_scoped_release rel;
                yolov8::ultralytics::loadWeights(self, vai::SafetensorsParser(p.c_str()));
            },
            nb::arg("path") = nb::none(),
            "Load weights from a safetensors file path (Ultralytics key naming). "
            "With no path, reads data/yolov8/<tier>/yolov8.safetensors."
        )
        .def(
            "set_profile_tags",
            [](yolov8::Model& self) { utils::setProfileTags(self); },
            "Tag every node for per-node GPU profiling. Tagging is what turns "
            "profiling on; tags may be set or changed at any point."
        );

    // ndarray in (zero-copy view; the STL caster copied per element) and
    // ndarray out (capsule-owned); processing runs with the GIL released.
    m.def("yolov8_preprocess",
          [](nb::ndarray<const uint8_t, nb::c_contig, nb::device::cpu> image,
             uint32_t width, uint32_t height, uint32_t target_size)
          {
              // bilinearResize indexes by width/height unchecked — a mismatch
              // with the actual byte count is an out-of-bounds read.
              if (width == 0 || height == 0 || target_size == 0 ||
                  image.size() != size_t(width) * height * 3)
                  throw nb::value_error(
                      "yolov8_preprocess: image must hold width*height*3 bytes (HWC RGB)");
              std::vector<float> result;
              {
                  nb::gil_scoped_release rel;
                  std::vector<uint8_t> vec(image.data(), image.data() + image.size());
                  result = preprocess(vec, width, height, target_size);
              }
              auto* chw = new std::vector<float>(std::move(result));
              nb::capsule owner(chw, [](void* p) noexcept {
                  delete static_cast<std::vector<float>*>(p);
              });
              size_t shape[1] = { chw->size() };
              return nb::ndarray<nb::numpy>(
                  chw->data(), 1, shape, owner, /*strides*/ nullptr, nb::dtype<float>());
          },
          nb::arg("image"), nb::arg("width"), nb::arg("height"),
          nb::arg("target_size") = 640u,
          "Letterbox-resize + normalize an HWC uint8 RGB image (flat numpy "
          "array) to a CHW float ndarray.");

    m.def("yolov8_postprocess",
          [](nb::ndarray<const float, nb::c_contig, nb::device::cpu> data,
             float conf_threshold, float iou_threshold, uint32_t image_size)
          {
              std::vector<Detection> dets;
              {
                  nb::gil_scoped_release rel;
                  std::vector<float> vec(data.data(), data.data() + data.size());
                  dets = postprocess(vec, conf_threshold, iou_threshold, image_size);
              }
              return dets;
          },
          nb::arg("data"), nb::arg("conf_threshold"), nb::arg("iou_threshold"), nb::arg("image_size"),
          "Confidence filter + NMS over the [4+numClasses, N] network output "
          "(flat float32 numpy array).");
}

}   // namespace pyvai
