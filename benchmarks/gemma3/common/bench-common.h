#ifndef BENCH_COMMON_H
#define BENCH_COMMON_H

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <string>
#include <vector>
#include <random>

namespace gemma3_bench {

inline constexpr int decodeTokens = 128;
inline constexpr int decodeRepetitions = 5;
inline constexpr int prefillRepetitions = 15;
inline constexpr int warmupPrefills = 3;
inline constexpr int warmupDecodeTokens = 8;
inline constexpr int runtimeContext = 8192;

inline constexpr uint32_t syntheticImageChannels = 3;
inline constexpr uint32_t syntheticImageSize = 896;
inline constexpr uint32_t randomSeed = 42;

inline constexpr std::array<int, 5> defaultTextLengths = { 128, 256, 512, 1024, 2048};

class BenchmarkArgumentParser
{
public:
    explicit BenchmarkArgumentParser(bool allowListDevices = false)
        : allowListDevices_(allowListDevices)
    {
    }

    bool parse(int argc, char** argv)
    {
        program_ = argc > 0 ? argv[0] : "benchmark";

        for (int i = 1; i < argc; ++i)
        {
            const std::string arg = argv[i];

            if (arg == "--showText") showText = true;
            else if (arg == "--list-devices" && allowListDevices_) listDevices = true;
            
            else if (arg == "--size")
            {
                if (++i == argc) return fail("--size needs a name");
                size = argv[i];
            }
            else if (arg == "--precision")
            {
                if (++i == argc) return fail("--precision needs a value");
                precision = argv[i];
            }
            else if (arg == "--imagepath")
            {
                if (++i == argc) return fail("--imagepath needs a path");
                imagepath = argv[i];
            }
            else if (arg == "--promptLengths")
            {
                if (i + 1 >= argc) return fail("--promptLengths needs at least one value");

                bool foundValue = false;

                while (i + 1 < argc)
                {
                    const std::string nextArg = argv[i + 1];

                    if (nextArg.rfind("--", 0) == 0) break;

                    ++i;

                    char* end = nullptr;
                    const long n = std::strtol(argv[i], &end, 10);

                    if (!end || *end || n <= 0)
                        return fail("invalid prompt length '" + std::string(argv[i]) + "'");

                    promptLengths.push_back(static_cast<int>(n));
                    foundValue = true;
                }

                if (!foundValue) return fail("--promptLengths needs at least one value");
            }
        }

        if (listDevices && argc != 2)
            return fail("--list-devices must be used alone");

        if (!isOneOf(size, modelSizes))
            return fail("unknown --size '" + size + "'");

        if (!isOneOf(precision, precisions))
            return fail("unknown --precision '" + precision + "'");


        if (promptLengths.empty())
            promptLengths.assign(defaultPromptLengths.begin(), defaultPromptLengths.end());

        return true;
    }

    void printErrorAndUsage() const
    {
        if (!error_.empty())
            std::fprintf(stderr, "error: %s\n", error_.c_str());

        std::fprintf(
            stderr,
            "usage: %s [--size NAME] [--precision fp16|fp32] [--imagepath PATH] "
            "[--promptLengths N ...] [--showText] [TEXT_TOKENS ...]%s\n",
            program_.c_str(),
            allowListDevices_ ? " | --list-devices" : "");

        std::fprintf(
            stderr,
            "  --size            checkpoint folder under data/gemma3/ "
            "(default: 4B), one of: 4B 1B\n");

        std::fprintf(
            stderr,
            "  --precision       weight, compute, and KV-cache dtype "
            "(default: fp16)\n");

        std::fprintf(stderr, "  --imagepath       input image path\n");

        std::fprintf(
            stderr,
            "  --promptLengths   one or more prompt token lengths "
            "(example: --promptLengths 128 256 512)\n");

        std::fprintf(
            stderr,
            "  --showText        accepted for compatibility; generated "
            "text is always printed\n");

        std::fprintf(
            stderr,
            "  TEXT_TOKENS       VQA text lengths "
            "(default: 128 256 512 1024 2048)\n");

        if (allowListDevices_)
            std::fprintf(
                stderr,
                "  --list-devices    list available llama.cpp devices and exit\n");
    }

    std::string size = "4B";
    std::string precision = "fp16";
    std::string imagepath = "";

    bool showText = true;
    bool listDevices = false;

    std::vector<int> promptLengths;

private:
    template<size_t N>
    static bool isOneOf(
        const std::string& value,
        const std::array<const char*, N>& values)
    {
        return std::any_of( values.begin(), values.end(),
            [&](const char* candidate){return value == candidate;});
    }

    bool fail(std::string message)
    {
        error_ = std::move(message);
        return false;
    }

    inline static constexpr std::array<const char*, 2> modelSizes = {
        "4B", "1B"
    };

    inline static constexpr std::array<const char*, 2> precisions = {
        "fp16", "fp32"
    };

    inline static constexpr std::array<int, 5> defaultPromptLengths = {
        128, 256, 512, 1024, 2048
    };

    bool allowListDevices_;
    std::string program_ = "benchmark";
    std::string error_;
};

inline constexpr const char* startOfImageToken = "<start_of_image>";
inline constexpr const char* endOfImageToken = "<end_of_image>";
inline constexpr const char* imageSoftToken = "<image_soft_token>";
inline constexpr const char* userPromptPrefix = "<bos><start_of_turn>user\n\n\n";
inline constexpr const char* textAfterImage = "\n\n";
inline constexpr const char* modelPromptSuffix =
    "<end_of_turn>\n<start_of_turn>model\n";

// Returns a deterministic, sufficiently long VQA-style question.  The same
// source text is tokenized by both benchmark backends before taking its first N
// tokens, just like the llama3 and gpt2 benchmarks.
std::string vqaPrompt()
{
    static constexpr const char* passage =
    "Study the image carefully and answer as a visual question-answering assistant. "

    "Describe the main subject, where it is located, what it is doing, and the direction "

    "in which it appears to be looking or moving. Identify nearby people, animals, vehicles, "

    "buildings, furniture, tools, signs, screens, plants, and other objects. Explain their "

    "colors, approximate sizes, shapes, materials, textures, and positions relative to one another. "

    "Note what is in the foreground, middle distance, background, upper and lower parts, and on "

    "the left and right sides. Discuss the lighting, shadows, reflections, weather, time of day, "

    "indoor or outdoor setting, camera viewpoint, depth, and visible perspective. Read any legible "

    "text without inventing missing letters. Distinguish direct observations from uncertain "

    "interpretations. Look for unusual details, partial occlusions, repeated patterns, possible "

    "hazards, and evidence of events that may have happened immediately before the captured moment. "

    "Compare similar objects and count them when the count is reasonably clear. If a detail cannot "

    "be determined from the image, state that limitation rather than guessing. Finally, combine "

    "the observations into a concise explanation of the scene and answer this question: where is "

    "the primary subject looking, and which visible evidence supports that conclusion?";

    std::string prompt;
    for (int i = 0; i < 24; ++i) prompt += passage;
    
    return prompt;
}

std::vector<uint8_t> makeRandom(const uint32_t srcW, const uint32_t srcH, const uint32_t channels)
{
    const size_t totalSize = static_cast<size_t>(srcW) * static_cast<size_t>(srcH) * static_cast<size_t>(channels);

    std::vector<uint8_t> output(totalSize);

    // Random seed
    std::random_device rd;
    std::mt19937 generator(rd());

    // Gaussian distribution
    // 평균: 127.5
    // 표준편차: 42.5
    std::normal_distribution<double> distribution(127.5, 42.5);

    for (auto& value : output)
    {
        double randomValue = distribution(generator);

        // 0 ~ 255 범위로 제한
        randomValue = std::clamp(randomValue, 0.0, 255.0);

        value = static_cast<uint8_t>(std::round(randomValue));
    }

    return output;
}

static double nowSec()
{
    using clock = std::chrono::steady_clock;
    return std::chrono::duration<double>(clock::now().time_since_epoch()).count();
}

struct Stats { double median, se; };
static Stats stats(std::vector<double> v)
{
    std::sort(v.begin(), v.end());
    const size_t n = v.size();
    const double median = n % 2 ? v[n / 2] : 0.5 * (v[n / 2 - 1] + v[n / 2]);
    double mean = 0.0;
    for (double x : v) mean += x;
    mean /= n;
    double var = 0.0;
    for (double x : v) var += (x - mean) * (x - mean);
    var = n > 1 ? var / (n - 1) : 0.0;
    return { median, std::sqrt(var / n) };
}

// Same table as the llama3 and gpt2 benches: request = text tokens asked for,
// tokens = what was actually prefilled (image and template tokens included).
// Prefill covers the vision encoder and the LLM prefill as one forward.
static void printHeader(int nGen, int nReps, int nPPReps)
{
    printf("\n decode %d tokens, %d reps · prefill %d reps · median ± SE\n", nGen, nReps, nPPReps);
    printf(" %7s  %-10s  %6s  %21s  %17s\n",
           "request", "framework", "tokens", "prefill (t/s)", "decode (t/s)");
    printf("────────────────────────────────────────────────────────────────\n");
}

static void printRow(const char* framework, int nReq, int nPrompt, Stats pp, Stats tg)
{
    printf(" %7d  %-10s  %6d  %11.2f ± %-7.2f  %9.2f ± %-6.2f\n",
           nReq, framework, nPrompt, pp.median, pp.se, tg.median, tg.se);
    fflush(stdout);
}

}

#endif //BENCH_COMMON_H
