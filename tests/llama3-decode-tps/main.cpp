// Llama 3 decode throughput: prefills a fixed prompt, then times nGen
// single-token forwards and reports tokens/s. The prefill that seeds each run
// is not counted, so the number is decode only.
//
//   test-llama3-decode-tps [nGen] [nReps]      default: 128 5

#include "models/llama3/model.h"
#include "models/llama3/meta-weights.h"
#include "vai/utils/tokenizer.h"
#include "eva-runtime.h"
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

using namespace vai;
using namespace vai::node;


double nowSec()
{
    using clock = std::chrono::steady_clock;
    return std::chrono::duration<double>(clock::now().time_since_epoch()).count();
}


int main(int argc, char** argv)
{
    const int nGen  = argc > 1 ? atoi(argv[1]) : 128;
    const int nReps = argc > 2 ? atoi(argv[2]) : 5;

    if (nGen <= 0 || nReps <= 0)
    {
        fprintf(stderr, "usage: %s [nGen] [nReps]\n", argv[0]);
        return 1;
    }

    eva::Device device = getVaiDevice();
    printf("device: vendorID 0x%04x, deviceID 0x%04x\n", device.vendorID(), device.deviceID());

    // TopKSampling draws via std::rand() each forward.
    srand(42);

    auto modelCfg = llama3::ModelConfig::_3_2_1B();
    printf("Loading Llama-%s model...\n", modelCfg.variant);
    llama3::Model model({
        .device = device,
        .wDefaultDtype = Dtype::Fp16,
    }, modelCfg, {
        .maxRuntimeContext = 4096,
    });

    loadWeights<"meta">(model);

    Tokenizer tokenizer(VAI_DATA_DIR "/llama3/tokenizer.json");

    const std::string prompt =
        "In recent decades, artificial intelligence has progressed from a niche "
        "academic curiosity to a force that quietly reshapes nearly every corner of "
        "modern life. ";
    const std::vector<int> promptTokens = tokenizer.encode(prompt, true);

    if ((int)promptTokens.size() + nGen > (int)model.runtimeCfg.maxRuntimeContext)
    {
        fprintf(stderr, "error: prompt %zu + %d generated exceeds the %u-token context\n",
                promptTokens.size(), nGen, model.runtimeCfg.maxRuntimeContext);
        return 1;
    }

    Tensor inTokens;
    Tensor outToken;

    auto prefill = [&]() {
        model.resetCache();
        inTokens = Tensor({(uint32_t)promptTokens.size()}).setData(promptTokens.data());
        model.prefill(inTokens, &outToken);
    };

    auto decode = [&]() -> double {
        const double t0 = nowSec();
        for (int gen = 0; gen < nGen; ++gen)
        {
            model.decode(&outToken);
            outToken.getData<uint32_t>();
        }
        return nGen / (nowSec() - t0);
    };

    // Shader compilation, buffer allocation and the GPU clock ramp are paid here.
    prefill();
    decode();

    printf("\n prompt %zu tokens · decode %d tokens · %d reps\n",
           promptTokens.size(), nGen, nReps);
    printf(" %4s  %10s  %10s\n", "rep", "t/s", "ms/token");
    printf("──────────────────────────────\n");

    std::vector<double> tps;
    for (int rep = 0; rep < nReps; ++rep)
    {
        prefill();
        const double v = decode();
        tps.push_back(v);
        printf(" %4d  %10.2f  %10.3f\n", rep, v, 1000.0 / v);
        fflush(stdout);
    }

    std::sort(tps.begin(), tps.end());
    const double median = nReps % 2 ? tps[nReps / 2]
                                    : 0.5 * (tps[nReps / 2 - 1] + tps[nReps / 2]);

    printf("──────────────────────────────\n");
    printf(" %4s  %10.2f  %10.3f\n", "med", median, 1000.0 / median);

    return 0;
}
