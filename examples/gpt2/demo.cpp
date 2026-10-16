/*=============================================================================
  Written by   oneway@vaistrata.com
  Task         GPT-2 — completes one prompt, profiles prefill and decode.
  Assets       data/gpt2/tokenizer.json
                 from  huggingface.co/openai-community/gpt2
=============================================================================*/

#include "models/gpt2/model.h"
#include "models/gpt2/openai-weights.h"
#include "vai/utils/tokenizer.h"
#include "vai/utils/time-checker.h"
#include "eva-runtime.h"
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

using namespace vai;


void setProfileTags(gpt2::Model& net)
{
    const uint32_t D = net.modelCfg.languageCfg.layerCfg.embedDim;
    const uint32_t V = net.modelCfg.languageCfg.vocabSize;

    auto lin = [](uint32_t in, uint32_t out) {
        return "Linear.[" + std::to_string(in) + "x" + std::to_string(out) + "]";
    };

    for (uint32_t i = 0; i < net.modelCfg.languageCfg.numLayers; ++i)
    {
        auto& b = net.decoder.layers[i];
        b.norm1.setProfileTag("LayerNorm");
        b.mhAttn.toQKV.setProfileTag(lin(D, 3 * D));
        b.mhAttn.attn.setProfileTag("Attn");
        b.mhAttn.toOut.setProfileTag(lin(D, D));
        b.norm2.setProfileTag("LayerNorm");
        b.ffn.fc1.setProfileTag(lin(D, 4 * D));
        b.ffn.fc2.setProfileTag(lin(4 * D, D));
    }
    net.decoder.finalNorm.setProfileTag("LayerNorm");
    net.lmHead.setProfileTag(lin(D, V));
    net.topkSampling.setProfileTag("Sampling");
}


int main()
{
    // Initialize device
    eva::Device device = getVaiDevice();

    // Load model
    auto modelCfg = gpt2::ModelConfig::_124M();
    printf("Loading GPT-2-%s model...\n", modelCfg.variant);
    gpt2::Model model({
        .device = device,
        .wDefaultDtype = Dtype::Fp16,
    }, modelCfg);

    // Configure the per-node GPU execution-time profiler
    setProfileTags(model);

    // Load weights
    printf("Loading weights...\n");
    {
        TimeChecker loadTimer("Weights loaded successfully in {}ms");
        loadWeights<"openai">(model);
    }

    // Load tokenizer
    Tokenizer tokenizer(VAI_DATA_DIR "/gpt2/tokenizer.json");

    // Prepare prompt and encode it to token ids
    std::string prompt =
        "In recent decades, artificial intelligence has progressed from a niche "
        "academic curiosity to a force that quietly reshapes nearly every corner of "
        "modern life. Algorithms now translate languages in real time, diagnose diseases "
        "from medical images, draft emails on our behalf, and pilot vehicles through "
        "unpredictable city streets. What once required entire research teams can "
        "sometimes be done by a single model running on a laptop. Yet despite these "
        "remarkable advances, the deeper questions remain unresolved: how should we "
        "measure understanding, what responsibilities accompany increasingly autonomous "
        "systems, and what role will human judgment play in a world where machines "
        "write, decide, and predict at unprecedented speed and scale? The next chapter "
        "of this story is still being written, and the choices we make in the coming "
        "years may";
    printf("\nPrompt: \"%s\"\n", prompt.c_str());
    std::vector<int> ids = tokenizer.encode(prompt);
    const uint32_t promptLength = (uint32_t) ids.size();

    // Initialize the input tensors
    Tensor promptTokenIds = Tensor({promptLength}).setData(ids.data());
    Tensor nextTokenId;

    // Warm-up
    {
        model.prefill(promptTokenIds, &nextTokenId);
        model.decode(&nextTokenId);
    }

    model.resetCache();
    model.resetProfileRecords();

    // Prefill
    {
        TimeChecker totalTimer("\n"
            "────────────────────────────────────────\n"
            "▶ Prefill {} tokens in {}ms \n"
            "────────────────────────────────────────", promptLength);

        model.prefill(promptTokenIds, &nextTokenId);
        ids.push_back(nextTokenId.getData<uint32_t>()[0]);
    }

    model.printProfile();

    // Decode
    {
        const int maxNewTokens = 128;

        TimeChecker totalTimer("\n"
            "────────────────────────────────────────\n"
            "▶ Decode {} tokens in {}ms \n"
            "────────────────────────────────────────", maxNewTokens);

        for (int gen = 0; gen < maxNewTokens; gen++)
        {
            model.decode(&nextTokenId);
            ids.push_back(nextTokenId.getData<uint32_t>()[0]);
        }
    }

    model.printProfile();

    // Print final generated text
    printf("\n=== Final Generated Text ===\n");
    std::string generatedText = tokenizer.decode(ids);
    printf("%s\n", generatedText.c_str());
    printf("\n--- Generation complete ---\n");

    return 0;
}
