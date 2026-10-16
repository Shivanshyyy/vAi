#ifndef YOLOX_PROCESSING_H
#define YOLOX_PROCESSING_H

#include "vai/utils/image-processor.h"
#include <cstdint>
#include <string>
#include <vector>

constexpr uint32_t NUM_COCO_CLASSES = 80u;
extern const char* const COCO_CLASSES[NUM_COCO_CLASSES];

struct Detection {
    float x1, y1, x2, y2;
    float confidence;
    uint32_t classId;
    std::string className;
};

std::vector<Detection> postprocess(
    const std::vector<float>& rawData,
    const ImageGeometry& geometry,
    float confThreshold,
    float iouThreshold,
    uint32_t numClasses = NUM_COCO_CLASSES);

#endif // YOLOX_PROCESSING_H
