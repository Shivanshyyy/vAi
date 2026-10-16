#ifndef PYVAI_UTILS_YOLOV8_PROCESSING_H
#define PYVAI_UTILS_YOLOV8_PROCESSING_H

// TODO: temporary copy of examples/yolov8/processing.h/.cpp, merged header-only
// — yolov8's pre/post steps live demo-side, but the bindings expose them
// (Detection, yolov8_preprocess/postprocess), so the code is copied instead
// of reaching into examples/. Verbatim except `inline` and the merge itself.
// Re-copy when the source changes; replace this file once the code has a
// shared home.

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

// COCO class names (80 classes)
constexpr uint32_t NUM_COCO_CLASSES = 80;
inline const char* const COCO_CLASSES[NUM_COCO_CLASSES] = {
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
    "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
    "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack",
    "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball",
    "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
    "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
    "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair",
    "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse",
    "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator",
    "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"
};


/////////////////////////////////////////////////////////////////////////////////////////
// Detection result
/////////////////////////////////////////////////////////////////////////////////////////
struct Detection {
    float x1, y1, x2, y2;
    float confidence;
    int   classId;
    std::string className;
};


/////////////////////////////////////////////////////////////////////////////////////////
// Image preprocessing - letterbox resize + normalize to CHW float
/////////////////////////////////////////////////////////////////////////////////////////

inline void bilinearResize(const uint8_t* src, uint32_t srcW, uint32_t srcH,
                           uint8_t* dst, uint32_t dstW, uint32_t dstH)
{
    float xRatio = static_cast<float>(srcW - 1) / std::max(dstW - 1, 1u);
    float yRatio = static_cast<float>(srcH - 1) / std::max(dstH - 1, 1u);

    for (uint32_t y = 0; y < dstH; ++y) {
        for (uint32_t x = 0; x < dstW; ++x) {
            float sx = x * xRatio;
            float sy = y * yRatio;

            uint32_t x1 = static_cast<uint32_t>(std::floor(sx));
            uint32_t y1 = static_cast<uint32_t>(std::floor(sy));
            uint32_t x2 = std::min(x1 + 1, srcW - 1);
            uint32_t y2 = std::min(y1 + 1, srcH - 1);

            float xDiff = sx - x1;
            float yDiff = sy - y1;

            for (int c = 0; c < 3; ++c) {
                float p11 = src[(y1 * srcW + x1) * 3 + c];
                float p21 = src[(y1 * srcW + x2) * 3 + c];
                float p12 = src[(y2 * srcW + x1) * 3 + c];
                float p22 = src[(y2 * srcW + x2) * 3 + c];

                float top    = p11 * (1.0f - xDiff) + p21 * xDiff;
                float bottom = p12 * (1.0f - xDiff) + p22 * xDiff;
                float value  = top * (1.0f - yDiff) + bottom * yDiff;

                dst[(y * dstW + x) * 3 + c] = static_cast<uint8_t>(value);
            }
        }
    }
}

inline void normalizeHWCtoCHW(const uint8_t* hwc, uint32_t width, uint32_t height, float* chw)
{
    for (uint32_t c = 0; c < 3; ++c)
        for (uint32_t y = 0; y < height; ++y)
            for (uint32_t x = 0; x < width; ++x)
                chw[c * height * width + y * width + x] =
                    static_cast<float>(hwc[(y * width + x) * 3 + c]) / 255.0f;
}

inline std::vector<float> preprocess(
    const std::vector<uint8_t>& image,
    uint32_t srcW,
    uint32_t srcH,
    uint32_t targetSize = 640)
{
    const uint8_t padValue = 114;

    float scale = std::min(
        static_cast<float>(targetSize) / srcW,
        static_cast<float>(targetSize) / srcH
    );
    uint32_t newW    = static_cast<uint32_t>(srcW * scale);
    uint32_t newH    = static_cast<uint32_t>(srcH * scale);
    uint32_t padLeft = (targetSize - newW) / 2;
    uint32_t padTop  = (targetSize - newH) / 2;

    std::vector<uint8_t> resized(newW * newH * 3);
    bilinearResize(image.data(), srcW, srcH, resized.data(), newW, newH);

    std::vector<uint8_t> padded(targetSize * targetSize * 3, padValue);
    for (uint32_t y = 0; y < newH; ++y) {
        uint32_t srcOff = y * newW * 3;
        uint32_t dstOff = ((padTop + y) * targetSize + padLeft) * 3;
        std::memcpy(&padded[dstOff], &resized[srcOff], newW * 3);
    }

    std::vector<float> chw(3 * targetSize * targetSize);
    normalizeHWCtoCHW(padded.data(), targetSize, targetSize, chw.data());

    return chw;
}


/////////////////////////////////////////////////////////////////////////////////////////
// Post-processing - confidence filter + NMS
/////////////////////////////////////////////////////////////////////////////////////////

inline float calculateIoU(const Detection& a, const Detection& b)
{
    float ix1 = std::max(a.x1, b.x1);
    float iy1 = std::max(a.y1, b.y1);
    float ix2 = std::min(a.x2, b.x2);
    float iy2 = std::min(a.y2, b.y2);

    float interW = std::max(0.0f, ix2 - ix1);
    float interH = std::max(0.0f, iy2 - iy1);
    float inter  = interW * interH;

    float areaA = (a.x2 - a.x1) * (a.y2 - a.y1);
    float areaB = (b.x2 - b.x1) * (b.y2 - b.y1);
    float uni   = areaA + areaB - inter;

    return (uni <= 0.0f) ? 0.0f : inter / uni;
}

inline std::vector<Detection> applyNMS(std::vector<Detection>& detections, float iouThreshold)
{
    std::sort(
        detections.begin(),
        detections.end(),
        [](auto& a, auto& b) { return a.confidence > b.confidence; }
    );

    std::vector<Detection> result;
    result.reserve(detections.size());
    std::vector<bool> suppressed(detections.size(), false);

    for (size_t i = 0; i < detections.size(); ++i)
    {
        if (suppressed[i])
            continue;
        result.push_back(detections[i]);

        for (size_t j = i + 1; j < detections.size(); ++j)
        {
            if (suppressed[j])
                continue;
            if (detections[i].classId != detections[j].classId)
                continue;
            if (calculateIoU(detections[i], detections[j]) > iouThreshold)
                suppressed[j] = true;
        }
    }

    return result;
}

// Input: [4 + numClasses, N] flat buffer — x1, y1, x2, y2, then per-class scores
inline std::vector<Detection> postprocess(
    const std::vector<float>& rawData,
    float confThreshold,
    float iouThreshold,
    uint32_t imageSize = 640u)
{
    // [4 + numClasses, N] row-major layout (e.g. [84, 8400] for COCO):
    //   row 0:               x1         (N floats)
    //   row 1:               y1         (N floats)
    //   row 2:               x2         (N floats)
    //   row 3:               y2         (N floats)
    //   row 4..4+numCls-1:   per-class sigmoid confidence (N floats each)
    constexpr uint32_t numClasses = NUM_COCO_CLASSES;
    constexpr uint32_t channels   = 4 + numClasses;
    const uint32_t N = rawData.size() / channels;

    const float* x1Row = rawData.data();
    const float* y1Row = x1Row + N;
    const float* x2Row = y1Row + N;
    const float* y2Row = x2Row + N;
    const float* clsBase = y2Row + N;        // [numClasses, N], row-major

    std::vector<Detection> detections;
    detections.reserve(256);
    for (uint32_t i = 0; i < N; ++i)
    {
        // argmax over class confidences for anchor i
        float bestConf = clsBase[i];
        uint32_t bestCls = 0;
        for (uint32_t c = 1; c < numClasses; ++c) {
            float v = clsBase[c * N + i];
            if (v > bestConf) { bestConf = v; bestCls = c; }
        }

        if (bestConf >= confThreshold)
        {
            Detection det;
            det.x1         = x1Row[i];
            det.y1         = y1Row[i];
            det.x2         = x2Row[i];
            det.y2         = y2Row[i];
            det.confidence = bestConf;
            det.classId    = bestCls;
            det.className  = COCO_CLASSES[bestCls];
            detections.push_back(det);
        }
    }

    detections = applyNMS(detections, iouThreshold);
    for (Detection& detection : detections)
    {
        detection.x1 = std::clamp(detection.x1, 0.0f, static_cast<float>(imageSize));
        detection.y1 = std::clamp(detection.y1, 0.0f, static_cast<float>(imageSize));
        detection.x2 = std::clamp(detection.x2, 0.0f, static_cast<float>(imageSize));
        detection.y2 = std::clamp(detection.y2, 0.0f, static_cast<float>(imageSize));
    }

    return detections;
}

#endif // PYVAI_UTILS_YOLOV8_PROCESSING_H
