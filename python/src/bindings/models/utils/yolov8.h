#ifndef PYVAI_UTILS_YOLOV8_H
#define PYVAI_UTILS_YOLOV8_H

// TODO: temporary copy of setProfileTags() from examples/yolov8/demo.cpp —
// the model members it walks are not exposed to Python, so the body is
// copied verbatim. Re-copy when the source changes; replace this file once
// a bindable path exists.

#include "models/yolov8/model.h"
#include <cstdint>
#include <string>

namespace pyvai::utils {
namespace {

using namespace vai;

void setProfileTags(yolov8::Model& net)
{
    const uint32_t imgSize = net.runtimeCfg.imageSize;
    const uint32_t H0 = imgSize;        // 640 (input image)
    const uint32_t H1 = imgSize / 2;    // 320 (p1   out)
    const uint32_t H2 = imgSize / 4;    // 160 (p2   out)
    const uint32_t H3 = imgSize / 8;    //  80 (p3   out)
    const uint32_t H4 = imgSize / 16;   //  40 (p4   out)
    const uint32_t H5 = imgSize / 32;   //  20 (p5   out)

    // Tag format: Op.[CixCo@H_in] — '.' builds a hierarchical tree (op family at top,
    // shape variants below).  For stride-2 convs H_in is the input spatial (output is
    // H_in/2);  for stride-1 / 1x1 convs H_in == H_out.
    auto fmt = [](const char* op, uint32_t Ci, uint32_t Co, uint32_t H) {
        return std::string(op) + ".[" + std::to_string(Ci) + "x" + std::to_string(Co)
             + "@" + std::to_string(H) + "]";
    };
    auto tag3x3s1 = [&](vai::node::Convolution& n, uint32_t H) {
        n.setProfileTag(fmt("Conv3x3s1", n.config_.inChannels, n.config_.outChannels, H));
    };
    auto tag3x3s2 = [&](vai::node::Convolution& n, uint32_t Hin) {
        n.setProfileTag(fmt("Conv3x3s2", n.config_.inChannels, n.config_.outChannels, Hin));
    };
    auto tag1x1bn = [&](vai::node::Convolution& n, uint32_t H) {
        n.setProfileTag(fmt("Conv1x1bn", n.config_.inChannels, n.config_.outChannels, H));
    };
    auto tag1x1 = [&](vai::node::Convolution& n, uint32_t H) {
        n.setProfileTag(fmt("Conv1x1",   n.config_.inChannels, n.config_.outChannels, H));
    };

    auto tagC2f = [&](yolov8::C2fModule& c2f, uint32_t H) {
        tag1x1bn(c2f.convIn,  H);
        tag1x1bn(c2f.convOut, H);
        for (auto& b : c2f.bottlenecks) {
            tag3x3s1(b.conv1, H);
            tag3x3s1(b.conv2, H);
        }
    };
    auto tagDS = [&](yolov8::Backbone::DownStage& ds, uint32_t Hin) {
        tag3x3s2(ds.conv, Hin);        // downsamples Hin -> Hin/2
        tagC2f  (ds.c2f,  Hin / 2);    // C2f runs at the downsampled spatial
    };
    auto tagSPPF = [&](yolov8::SPPFModule& s, uint32_t H) {
        tag1x1bn(s.convIn,  H);
        tag1x1bn(s.convOut, H);
    };
    auto tagPredTail = [&](yolov8::Head::PredictorTail& tail, uint32_t H) {
        tag3x3s1(tail.conv2, H);
        tag1x1  (tail.pred,  H);
    };
    auto tagPredPair = [&](yolov8::Head::PredictorPair& pair, uint32_t H) {
        tag3x3s1(pair.conv1, H);
        tagPredTail(pair.reg, H);
        tagPredTail(pair.cls, H);
    };

    // Backbone — pass each stage its INPUT spatial; stride-2 convs halve it.
    auto& bb = net.backbone;
    tag3x3s2(bb.p1, H0);
    tagDS(bb.p2,    H1);
    tagDS(bb.p3,    H2);
    tagDS(bb.p4,    H3);
    tagDS(bb.p5Pre, H4);
    tagSPPF(bb.p5,  H5);

    // Neck (spatial follows the target p# scale of each branch)
    auto& nk = net.neck;
    tagC2f  (nk.temp_p4, H4);
    tagC2f  (nk.out_p3,  H3);
    tag3x3s2(nk.conv1,   H3);   // p3 -> p4 downsample (80 -> 40)
    tagC2f  (nk.out_p4,  H4);
    tag3x3s2(nk.conv2,   H4);   // p4 -> p5 downsample (40 -> 20)
    tagC2f  (nk.out_p5,  H5);

    // Head — three detection heads at p3 / p4 / p5
    auto& hd = net.head;
    tagPredPair(hd.p3, H3);
    tagPredPair(hd.p4, H4);
    tagPredPair(hd.p5, H5);
}

}   // namespace
}   // namespace pyvai::utils

#endif  // PYVAI_UTILS_YOLOV8_H
