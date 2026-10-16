/*=============================================================================
  gemma3 layer dump (vAi side)

  Runs the Gemma 3 4B VLM exactly as examples/gemma3/demo-4B.cpp does, with two
  changes that make the run comparable against PyTorch:

    * greedy sampling (topkSampling k = 1), so no draw enters the comparison
    * every layer boundary is tapped with an extra graph output and written to
      <out>/<name>.npy

  The image is read from the PyTorch dump (pixel_values.npy) when it is there,
  so the two sides start from the same pixels and the tables measure the model
  rather than the resize. Without it the demo's own preprocessing runs and the
  header says so.

  Decode is teacher-forced on <torch>/steps.ids.npy when present: both sides
  consume the same tokens, so a step's logit gap is that step's own error and
  not the tail of an earlier divergence. Each step also records what this side
  would have picked, which is what dates the first flip.

    test-gemma3-dump [--steps N] [--out DIR] [--torch DIR] [--no-vision] [--fp32]
=============================================================================*/

#include "models/gemma3/model.h"
#include "models/gemma3/google-weights.h"
#include "vai/utils/image-processor.h"
#include "vai/utils/tokenizer.h"
#include "npy.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <stdexcept>
#include <string>
#include <vector>

using namespace vai;


// The prompt of examples/gemma3/demo-4B.cpp, byte for byte: the chat template
// with exactly tokensPerImage placeholder ids between BOI and EOI.
static std::vector<int> makePrompt(
    const Tokenizer& tokenizer, const gemma3::ModelConfig& cfg, const std::string& question)
{
    auto requireToken = [&](const char* piece) {
        int id = tokenizer.pieceToId(piece);
        if (id < 0) throw std::runtime_error(std::string("tokenizer has no token ") + piece);
        return id;
    };
    if (!cfg.visionCfg) throw std::runtime_error("checkpoint has no vision configuration");

    const int boi = requireToken("<start_of_image>");
    const int eoi = requireToken("<end_of_image>");
    const int image = requireToken("<image_soft_token>");
    if (uint32_t(image) != cfg.visionCfg->imageTokenId)
        throw std::runtime_error("tokenizer image token does not match ModelConfig");

    std::vector<int> tokens = tokenizer.encode("<bos><start_of_turn>user\n\n\n");
    tokens.push_back(boi);
    tokens.insert(tokens.end(), cfg.visionCfg->tokensPerImage, image);
    tokens.push_back(eoi);
    std::vector<int> suffix = tokenizer.encode(
        "\n\n" + question + "<end_of_turn>\n<start_of_turn>model\n");
    tokens.insert(tokens.end(), suffix.begin(), suffix.end());
    return tokens;
}


// ---------------------------------------------------------------------------
// Tap readback: a tap that this forward never reached comes back empty, and
// fp16 tensors are widened here so every dump is float32.
// ---------------------------------------------------------------------------
static float halfToFloat(uint16_t h)
{
    const uint32_t sign = uint32_t(h & 0x8000u) << 16;
    const uint32_t exp = (h >> 10) & 0x1Fu;
    const uint32_t man = h & 0x3FFu;
    uint32_t bits;
    if (exp == 0)
        bits = man ? [&] {                     // subnormal: renormalize
            uint32_t e = 0, m = man;
            while (!(m & 0x400u)) { m <<= 1; ++e; }
            m &= 0x3FFu;
            return sign | ((127 - 15 - e) << 23) | (m << 13);
        }() : sign;
    else if (exp == 31)
        bits = sign | 0x7F800000u | (man << 13);
    else
        bits = sign | ((exp + 127 - 15) << 23) | (man << 13);

    float f;
    memcpy(&f, &bits, 4);
    return f;
}

struct Dump
{
    std::string dir;
    int written = 0;

    void operator()(const std::string& name, Tensor& t)
    {
        if (!t.validShape() || t.numElements() == 0)
            return;                            // tap not on this forward's path

        const size_t n = t.numElements();
        std::vector<float> host(n);
        if (t.dtype() == Dtype::Fp16)
        {
            const uint16_t* src = t.getData<uint16_t>();
            for (size_t i = 0; i < n; ++i) host[i] = halfToFloat(src[i]);
        }
        else
        {
            memcpy(host.data(), t.getData<float>(), n * sizeof(float));
        }
        npy::write(dir + "/" + name + ".npy", t.shape(), host.data());
        ++written;
    }
};


int main(int argc, char** argv)
{
    int steps = 64;
    std::string outDir = "tests/gemma3/dumps/vai";
    std::string torchDir = "tests/gemma3/dumps/torch_32";
    bool withVision = true;
    bool fp32 = false;

    for (int i = 1; i < argc; ++i)
    {
        if (!strcmp(argv[i], "--steps") && i + 1 < argc)      steps = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--out") && i + 1 < argc)   outDir = argv[++i];
        else if (!strcmp(argv[i], "--torch") && i + 1 < argc) torchDir = argv[++i];
        else if (!strcmp(argv[i], "--no-vision"))             withVision = false;
        else if (!strcmp(argv[i], "--fp32"))                  fp32 = true;
        else
        {
            fprintf(stderr, "usage: %s [--steps N] [--out DIR] [--torch DIR] [--no-vision] [--fp32]\n", argv[0]);
            return 1;
        }
    }
    std::filesystem::create_directories(outDir);

    const std::string question = "Where is he looking?";

    eva::Device device = getVaiDevice();
    auto modelCfg = gemma3::ModelConfig::_4B();
    // --fp32: weights, compute and K/V all at fp32, so the gap to PyTorch fp32
    // is the implementation's own and not the fp16 rounding.
    const Dtype dt = fp32 ? Dtype::Fp32 : Dtype::Fp16;
    printf("dtype  : %s\n", fp32 ? "fp32" : "fp16");
    gemma3::Model model({
        .device = device,
        .wDefaultDtype = dt,
    }, modelCfg, {.kvCacheDtype = dt, .maxRuntimeContext = 4096});

    printf("loading weights...\n");
    loadWeights<"google">(model);
    model.topkSampling.config_.k = 1;          // greedy, so no draw enters the comparison

    // -----------------------------------------------------------------------
    // Taps. An output slot fans out, so these ride alongside the real edges and
    // change nothing the graph computes. Names match the PyTorch module they
    // are compared against (see dump_torch.py).
    // -----------------------------------------------------------------------
    std::vector<std::pair<std::string, Tensor>> taps;
    auto tap = [&](const std::string& name, auto& producer) {
        producer - model.output(name);
        taps.emplace_back(name, Tensor{});
    };

    const auto& v = *modelCfg.visionCfg;
    auto& vision = model.visionTower.value();
    tap("vis.patches", vision.patchEmbed.mergePatches);        // pixels cut into (N, C*P*P) rows
    tap("vis.embed", vision.patchEmbed);                       // patch conv + position table
    if (withVision)                                            // ~0.85 GB of the dump
        for (uint32_t i = 0; i < v.numLayers; ++i)
        {
            char name[32];
            snprintf(name, sizeof(name), "vis.%02u", i);
            tap(name, vision.encoder.layers[i]);
        }
    tap("vis.post_norm", vision.encoder.finalNorm);
    tap("vis.pool", vision.toTokens);                          // 64x64 patches -> 16x16 tokens
    tap("vis.pool_norm", vision.imageNorm);
    tap("vis.proj", vision.projector);                         // -> text embedDim

    tap("txt.embed", model.imageScatter);       // multimodal rows replace placeholders
    tap("txt.embed_text", model.embedding);                    // the decode path's embedding
    for (uint32_t i = 0; i < modelCfg.languageCfg.numLayers; ++i)
    {
        char name[32];
        snprintf(name, sizeof(name), "txt.%02u", i);
        tap(name, model.decoder.layers[i]);
    }
    tap("txt.final_norm", model.decoder.finalNorm);
    tap("logits", model.lmHead);

    Tokenizer tokenizer(VAI_DATA_DIR "/gemma3/4B/tokenizer.json");
    std::vector<int> tokens = makePrompt(tokenizer, modelCfg, question);
    const uint32_t promptLength = (uint32_t)tokens.size();

    // -----------------------------------------------------------------------
    // Pixels: the PyTorch dump when it exists, so both sides see one image.
    // -----------------------------------------------------------------------
    std::vector<float> chw;
    const std::string pixPath = torchDir + "/pixel_values.npy";
    if (std::filesystem::exists(pixPath))
    {
        std::vector<uint32_t> shape;
        chw = npy::read<float>(pixPath, shape);
        size_t want = size_t(v.imageChannels) * v.imageSize * v.imageSize;
        if (chw.size() != want)
            throw std::runtime_error("pixel_values.npy has " + std::to_string(chw.size())
                                     + " elements, expected " + std::to_string(want));
        printf("pixels : %s (shared with PyTorch)\n", pixPath.c_str());
    }
    else
    {
        chw = ImageProcessor(VAI_DATA_DIR "/gemma3/image.jpg")
            .resize(v.imageSize, Resize::BilinearAntialiasInt16)
            .normalize(-1.0f, 1.0f)
            .chw();
        printf("pixels : demo preprocessing — run dump_torch.py first to share one image\n");
    }

    // Teacher-forcing tokens: the PyTorch run's own picks, so a step's inputs
    // are identical on both sides even where the picks differ.
    std::vector<int32_t> forced;
    const std::string forcedPath = torchDir + "/steps.ids.npy";
    if (std::filesystem::exists(forcedPath))
    {
        std::vector<uint32_t> shape;
        forced = npy::read<int32_t>(forcedPath, shape);
        printf("decode : teacher-forced on %s (%zu ids)\n", forcedPath.c_str(), forced.size());
    }
    else
    {
        printf("decode : free-running (no steps.ids.npy)\n");
    }

    printf("prompt : %u tokens\n", promptLength);
    {
        std::vector<int32_t> ids(tokens.begin(), tokens.end());
        npy::write(outDir + "/prompt.ids.npy", {promptLength}, ids.data(), true);
    }

    Tensor inImage = Tensor({v.imageChannels, v.imageSize, v.imageSize}).setFloats(std::move(chw));
    Tensor inTokens = Tensor({promptLength}).setData(tokens.data());
    Tensor outToken;

    // ----- image prefill ---------------------------------------------------
    model.resetCache();
    model.prefill(inImage, inTokens, &outToken);

    int tok = (int)outToken.getData<uint32_t>()[0];
    printf("prefill: argmax %d\n", tok);

    // ----- decode ----------------------------------------------------------
    // Per step: the logits, plus what each side picked. The step's input is the
    // PyTorch token when teacher-forcing, so the rows stay comparable even after
    // the first disagreement.
    std::vector<int32_t> picks;
    std::vector<int32_t> fed;
    inTokens = Tensor({1});

    for (int s = 0; s < steps; ++s)
    {
        int in = (s < (int)forced.size()) ? forced[s] : tok;
        fed.push_back(in);

        outToken = Tensor({1}).setData(&in);
        model.decode(&outToken);
        tok = (int)outToken.getData<uint32_t>()[0];
        picks.push_back(tok);
    }

    npy::write(outDir + "/steps.fed.npy", {(uint32_t)fed.size()}, fed.data(), true);
    npy::write(outDir + "/steps.picks.npy", {(uint32_t)picks.size()}, picks.data(), true);

    printf("decode : %d steps\n", steps);
    printf("wrote  : token arrays under %s\n", outDir.c_str());
    return 0;
}
