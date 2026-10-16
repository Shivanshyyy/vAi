#pragma once

#include "benchmark.hpp"

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

struct Detection
{
    float x1;
    float y1;
    float x2;
    float y2;
    float confidence;
    uint32_t classId;
};

namespace yolox_benchmark {

inline constexpr uint32_t kInputChannels = 3;
inline constexpr uint32_t kInputHeight = 640;
inline constexpr uint32_t kInputWidth = 640;
inline constexpr uint32_t kInputSize = kInputChannels * kInputHeight * kInputWidth;
inline constexpr uint32_t kCocoClassCount = 80;
inline constexpr uint32_t kOutputAnchors = 8400;
inline constexpr uint32_t kOutputRows = 5 + kCocoClassCount;
inline constexpr uint32_t kOutputSize = kOutputRows * kOutputAnchors;
inline constexpr float kConfidenceThreshold = 0.25f;
inline constexpr float kIouThreshold = 0.45f;

std::vector<uint16_t> floatsToHalfBits(const std::vector<float>& values);
std::vector<float> halfBitsToFloats(const uint16_t* data, size_t count);

std::vector<float> preprocessImage(const std::string& path);

void validateModelOutputSize(size_t outputSize, const std::string& runtimeName);
std::vector<Detection> postprocessOutput(const std::vector<float>& output);
void printDetectionSummary(const std::vector<Detection>& detections);

} // namespace yolox_benchmark
