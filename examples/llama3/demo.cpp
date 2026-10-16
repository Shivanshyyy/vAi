/*=============================================================================
  Written by   oneway@vaistrata.com
  Task         Llama 3 — completes one prompt, profiles prefill and decode.
  Assets       data/llama3/tokenizer.json
                 from  huggingface.co/meta-llama/Llama-3.2-1B-Instruct
                 or  huggingface.co/unsloth/Llama-3.2-1B-Instruct
=============================================================================*/

#include "models/llama3/model.h"
#include "models/llama3/meta-weights.h"
#include "vai/utils/tokenizer.h"
#include "vai/utils/time-checker.h"
#include "eva-runtime.h"
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

using namespace vai;


void setProfileTags(llama3::Model& net)
{
    const auto& layerCfg = net.modelCfg.languageCfg.layerCfg;
    const uint32_t D    = layerCfg.embedDim;
    const uint32_t Dkv  = layerCfg.headDim * layerCfg.numKVHeads;
    const uint32_t Dff  = layerCfg.ffnHiddenDim;
    const uint32_t V    = net.modelCfg.languageCfg.vocabSize;

    auto lin = [](uint32_t in, uint32_t out) {
        return "Linear.[" + std::to_string(in) + "x" + std::to_string(out) + "]";
    };

    for (uint32_t i = 0; i < net.modelCfg.languageCfg.numLayers; ++i)
    {
        auto& b = net.decoder.layers[i];
        b.norm1.setProfileTag("RmsNorm");
        b.mhAttn.toQ.setProfileTag(lin(D, D));
        b.mhAttn.toK.setProfileTag(lin(D, Dkv));
        b.mhAttn.toV.setProfileTag(lin(D, Dkv));
        b.mhAttn.ropeQ->setProfileTag("Rope");
        b.mhAttn.ropeK->setProfileTag("Rope");
        b.mhAttn.attn.setProfileTag("Attn");
        b.mhAttn.toOut.setProfileTag(lin(D, D));
        b.norm2.setProfileTag("RmsNorm");
        b.ffn.gateValue.setProfileTag(lin(D, 2 * Dff));
        b.ffn.gatedMul.setProfileTag("GatedMul");
        b.ffn.out.setProfileTag(lin(Dff, D));
    }
    net.decoder.finalNorm.setProfileTag("RmsNorm");
    net.lmHead.setProfileTag(lin(D, V));
    net.topkSampling.setProfileTag("Sampling");
}


int main()
{
    // Initialize device
    eva::Device device = getVaiDevice();

    // Load model
    auto modelCfg = llama3::ModelConfig::_3_2_1B();
    printf("Loading Llama-%s model...\n", modelCfg.variant);
    llama3::Model model({
        .device = device,
        .wDefaultDtype = Dtype::Fp16,
    }, modelCfg, {
        .maxRuntimeContext = 4096,
    });

    // Configure the per-node GPU execution-time profiler
    setProfileTags(model);

    // Load weights
    printf("Loading weights...\n");
    {
        TimeChecker loadTimer("Weights loaded successfully in {}ms");
        loadWeights<"meta">(model);
    }

    // Load tokenizer
    Tokenizer tokenizer(VAI_DATA_DIR "/llama3/tokenizer.json");

    // Prepare prompt and encode it to token ids
    std::string prompt =
        "In recent decades, artificial intelligence has progressed from a niche "
        "academic curiosity to a force that quietly reshapes nearly every corner of "
        "modern life. "
        // "Algorithms now translate languages in real time, diagnose diseases "
        // "from medical images, draft emails on our behalf, and pilot vehicles through "
        // "unpredictable city streets. What once required entire research teams can "
        // "sometimes be done by a single model running on a laptop. Yet despite these "
        // "remarkable advances, the deeper questions remain unresolved: how should we "
        // "measure understanding, what responsibilities accompany increasingly autonomous "
        // "systems, and what role will human judgment play in a world where machines "
        // "write, decide, and predict at unprecedented speed and scale? The next chapter "
        // "of this story is still being written, and the choices we make in the coming "
        // "years may"
        ;
    printf("\nPrompt: \"%s\"\n", prompt.c_str());
    std::vector<int> ids = tokenizer.encode(prompt, true);
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
