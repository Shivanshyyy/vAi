// Chunked prefill on the llama3 sample model: one prompt prefilled at several
// chunk sizes, whose generated ids must all match the unchunked run.
//
//   test-chunked-prefill [nGen]      default: 8

#include "models/llama3/model.h"
#include "models/llama3/meta-weights.h"
#include "vai/utils/tokenizer.h"
#include "eva-runtime.h"
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

using namespace vai;


double nowSec()
{
    using clock = std::chrono::steady_clock;
    return std::chrono::duration<double>(clock::now().time_since_epoch()).count();
}


// Prefills in forwards of `chunk` tokens, 0 for the whole prompt in one, and
// times it to the end of the last.
double prefill(llama3::Model& model, const std::vector<int>& prompt,
               uint32_t chunk, Tensor& outToken)
{
    model.resetCache();
    model.runtimeCfg.prefillChunk = chunk;

    Tensor inTokens = Tensor({(uint32_t)prompt.size()}).setData(prompt.data());

    const double t0 = nowSec();
    model.prefill(inTokens, &outToken);
    model.sync();
    return (nowSec() - t0) * 1000.0;
}


std::vector<int> decode(llama3::Model& model, Tensor& outToken, int nGen)
{
    std::vector<int> tokens;
    int token = (int)outToken.getData<uint32_t>()[0];

    for (int gen = 0; gen < nGen; ++gen)
    {
        tokens.push_back(token);
        model.decode(&outToken);
        token = (int)outToken.getData<uint32_t>()[0];
    }
    return tokens;
}


int main(int argc, char** argv)
{
    const int nGen = argc > 1 ? atoi(argv[1]) : 8;

    if (nGen <= 0)
    {
        fprintf(stderr, "usage: %s [nGen]\n", argv[0]);
        return 1;
    }

    eva::Device device = getVaiDevice();
    printf("device: vendorID 0x%04x, deviceID 0x%04x\n", device.vendorID(), device.deviceID());

    auto modelCfg = llama3::ModelConfig::_3_2_1B();
    printf("Loading Llama-%s model...\n", modelCfg.variant);
    llama3::Model model({
        .device = device,
        .wDefaultDtype = Dtype::Fp16,
    }, modelCfg, {
        .maxRuntimeContext = 4096,
        .prefillChunk = 16,      // the stack builds the gate only for a chunked model
    });

    loadWeights<"meta">(model);

    // Greedy, so a row that differs here differs in the logits rather than in
    // the uniform TopKSampling draws, which the chunk count would also shift.
    model.topkSampling.config_.k = 1;

    Tokenizer tokenizer(VAI_DATA_DIR "/llama3/tokenizer.json");

    const std::string paragraph =
        "In recent decades, artificial intelligence has progressed from a niche "
        "academic curiosity to a force that quietly reshapes nearly every corner of "
        "modern life. ";

    const std::vector<int> prompt = tokenizer.encode(paragraph, true);

    Tensor outToken;

    // Shader compilation, buffer allocation and the GPU clock ramp are paid here.
    prefill(model, prompt, 0, outToken);
    decode(model, outToken, nGen);

    printf("\n prompt %zu tokens · %d generated\n", prompt.size(), nGen);
    printf(" %7s  %9s  %s\n", "chunk", "prefill", "generated ids");
    printf("──────────────────────────────────────────────────\n");

    std::vector<int> reference;
    int mismatches = 0;

    for (uint32_t c : {0u, 1u, 3u, 8u, 16u, 32u})
    {
        const double ms = prefill(model, prompt, c, outToken);
        const std::vector<int> tokens = decode(model, outToken, nGen);

        if (reference.empty())
            reference = tokens;

        printf(" %7u  %7.1fms  ", c ? c : (uint32_t)prompt.size(), ms);
        for (int t : tokens)
            printf("%d ", t);
        if (tokens != reference)
        {
            printf(" MISMATCH");
            ++mismatches;
        }
        printf("\n");
        fflush(stdout);
    }

    if (mismatches)
        printf("\n%d chunk size(s) generated a different token stream\n", mismatches);

    return mismatches ? 1 : 0;
}
