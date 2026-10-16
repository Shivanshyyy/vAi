/*=============================================================================
  Written by   phg9296@vaistrata.com
  Task         YOLOX — detects objects in one image and reports timings.
  Assets       data/yolox/dog.jpg
                 from  github.com/Megvii-BaseDetection/YOLOX/tree/main/assets
=============================================================================*/

#include "models/yolox/megvii-weights.h"
#include "models/yolox/model.h"
#include "processing.h"
#include "vai/utils/image-processor.h"
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

using namespace vai;

namespace {

void setProfileTags(yolox::Model& model)
{
    auto tagConv = [](vai::node::Convolution& conv, uint32_t spatial) {
        const auto& config = conv.config_;
        conv.setProfileTag(
            "Conv" + std::to_string(config.kernelSize) + "x"
            + std::to_string(config.kernelSize) + "s" + std::to_string(config.stride)
            + ".[" + std::to_string(config.inChannels) + "x"
            + std::to_string(config.outChannels) + "@" + std::to_string(spatial) + "]");
    };
    auto tagBottleneck = [&](yolox::Bottleneck& block, uint32_t spatial) {
        tagConv(block.conv1, spatial);
        tagConv(block.conv2, spatial);
    };
    auto tagCsp = [&](yolox::CSPLayer& csp, uint32_t spatial) {
        tagConv(csp.conv1, spatial);
        tagConv(csp.conv2, spatial);
        for (auto& block : csp.bottlenecks)
            tagBottleneck(block, spatial);
        tagConv(csp.conv3, spatial);
    };
    auto tagBranch = [&](yolox::Head::PredictionBranch& branch, uint32_t spatial) {
        tagConv(branch.stem, spatial);
        tagConv(branch.clsConv1, spatial);
        tagConv(branch.clsConv2, spatial);
        tagConv(branch.clsPred, spatial);
        tagConv(branch.regConv1, spatial);
        tagConv(branch.regConv2, spatial);
        tagConv(branch.regObjPred, spatial);
    };

    const uint32_t h = model.runtimeCfg.imageSize;
    auto& backbone = model.backbone;
    backbone.stem.conv.setProfileTag(
        "FocusConv3x3s1.[3x" +
        std::to_string(backbone.stem.conv.config_.outChannels) + "@" +
        std::to_string(h / 2u) + "]");
    tagConv(backbone.dark2.conv, h / 2u);
    tagCsp(backbone.dark2.csp, h / 4u);
    tagConv(backbone.dark3.conv, h / 4u);
    tagCsp(backbone.dark3.csp, h / 8u);
    tagConv(backbone.dark4.conv, h / 8u);
    tagCsp(backbone.dark4.csp, h / 16u);
    tagConv(backbone.dark5.conv, h / 16u);
    tagConv(backbone.dark5.spp.conv1, h / 32u);
    backbone.dark5.spp.sppf.setProfileTag("SPPF");
    tagConv(backbone.dark5.spp.conv2, h / 32u);
    tagCsp(backbone.dark5.csp, h / 32u);

    auto& neck = model.neck;
    tagConv(neck.lateralConv0, h / 32u);
    tagCsp(neck.c3P4, h / 16u);
    tagConv(neck.reduceConv1, h / 16u);
    tagCsp(neck.c3P3, h / 8u);
    tagConv(neck.bottomUpConv2, h / 8u);
    tagCsp(neck.c3N3, h / 16u);
    tagConv(neck.bottomUpConv1, h / 16u);
    tagCsp(neck.c3N4, h / 32u);

    tagBranch(model.head.p3, h / 8u);
    tagBranch(model.head.p4, h / 16u);
    tagBranch(model.head.p5, h / 32u);
    model.detector.decode.setProfileTag("Decode");
}

} // namespace

int main()
{
    constexpr yolox::Tier tier = yolox::Tier::LARGE;
    const std::string imagePath = VAI_DATA_DIR "/yolox/dog.jpg";

    eva::Device device = getVaiDevice();
    const yolox::ModelConfig cfg = yolox::getConfigCOCO(tier);
    const yolox::RuntimeConfig runtimeCfg {
        .imageSize = 640u,
        .confThreshold = 0.25f,
        .iouThreshold = 0.45f,
    };
    yolox::Model model({
        .device = device,
        .wDefaultDtype = Dtype::Fp16,
    }, cfg, runtimeCfg);

    std::printf("Loading YOLOX-%s weights...\n", yolox::tierName(tier));
    loadWeights<"megvii">(model);
    setProfileTags(model);

    ImageProcessor image(imagePath);
    image.letterbox(runtimeCfg.imageSize, Align::TopLeft, 114.0f).bgr().chw();
    Tensor input = Tensor({3u, runtimeCfg.imageSize, runtimeCfg.imageSize})
        .setFloats(image);
    Tensor output;

    for (uint32_t i = 0; i < 5u; ++i)
        model({{"in0", input}}, {{"out0", &output}});
    model.resetProfileRecords();

    for (uint32_t i = 0; i < 10u; ++i)
        model({{"in0", input}}, {{"out0", &output}});
    model.printProfile();
    std::vector<float> raw(output.numElements());
    std::memcpy(raw.data(), output.getData(), raw.size() * sizeof(float));
    std::vector<Detection> detections = postprocess(
        raw, image.geometry(),
        runtimeCfg.confThreshold, runtimeCfg.iouThreshold, cfg.numClasses);

    std::printf("\n=== Detections (%zu) ===\n", detections.size());
    for (size_t i = 0; i < detections.size(); ++i)
    {
        const Detection& detection = detections[i];
        std::printf(
            "[%zu] %s (%.2f%%): [%.1f, %.1f, %.1f, %.1f]\n",
            i, detection.className.c_str(), detection.confidence * 100.0f,
            detection.x1, detection.y1, detection.x2, detection.y2);
    }
    return 0;
}
