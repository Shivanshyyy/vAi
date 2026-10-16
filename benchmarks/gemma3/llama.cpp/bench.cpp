#include "llama.h"
#include "ggml-backend.h"
#include "mtmd.h"
#include "bench-common.h"
#include "vai/utils/stb_image.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

using namespace gemma3_bench;

[[noreturn]] static void fail(const char* msg)
{
    fprintf(stderr, "error: %s\n", msg);
    exit(1);
}

static void nullLog(ggml_log_level, const char*, void*) {}

static std::string tokensToString(const llama_vocab* vocab,
                                  const std::vector<llama_token>& tokens,
                                  int count)
{
    std::string text;
    std::vector<char> piece(256);
    for (int i = 0; i < count; ++i)
    {
        int n = llama_token_to_piece(vocab, tokens[i], piece.data(),
                                     static_cast<int32_t>(piece.size()), 0, false);
        if (n < 0)
        {
            piece.resize(static_cast<size_t>(-n));
            n = llama_token_to_piece(vocab, tokens[i], piece.data(),
                                     static_cast<int32_t>(piece.size()), 0, false);
        }
        if (n > 0) text.append(piece.data(), static_cast<size_t>(n));
    }
    return text;
}

static std::string visionPrompt(const char* imageMarker, const std::string& question)
{
    return std::string(gemma3_bench::userPromptPrefix) + imageMarker
        + gemma3_bench::textAfterImage + question + gemma3_bench::modelPromptSuffix;
}

int main(int argc, char** argv)
{
    if (!getenv("BENCH_VERBOSE"))
    {
        llama_log_set(nullLog, nullptr);
        mtmd_log_set(nullLog, nullptr);
    }

    constexpr int nGen = gemma3_bench::decodeTokens;
    constexpr int nReps = gemma3_bench::decodeRepetitions;
    constexpr int nPPReps = gemma3_bench::prefillRepetitions;
    constexpr int nWarmPre = gemma3_bench::warmupPrefills;
    constexpr int nWarmGen = gemma3_bench::warmupDecodeTokens;
    constexpr int nCtx = gemma3_bench::runtimeContext;
    constexpr uint32_t seed = gemma3_bench::randomSeed;

    gemma3_bench::BenchmarkArgumentParser args(true);
    if (!args.parse(argc, argv))
    {
        args.printErrorAndUsage();
        return 1;
    }

    if (args.listDevices)
    {
        ggml_backend_load_all();
        llama_backend_init();
        for (size_t i = 0; i < ggml_backend_dev_count(); ++i)
        {
            ggml_backend_dev_t d = ggml_backend_dev_get(i);
            printf("[DEV %zu] %s — %s\n", i, ggml_backend_dev_name(d), ggml_backend_dev_description(d));
        }
        llama_backend_free();
        return 0;
    }

    enum ggml_type cacheType;
    const char* filePrecision = nullptr;
    if (args.precision == "fp16") { cacheType = GGML_TYPE_F16; filePrecision = "f16"; }
    else { cacheType = GGML_TYPE_F32; filePrecision = "f32"; }

    const bool hasVision = args.size == "4B";
    const std::string dataDir = std::string(VAI_DATA_DIR "/gemma3/") + args.size + "/" + "GGUF/";
    const std::string modelPath = dataDir + "model-" + filePrecision + ".gguf";
    const std::string mmprojPath = dataDir + "mmproj-" + filePrecision + ".gguf";

    ggml_backend_load_all();
    llama_backend_init();
    const int nDev = static_cast<int>(ggml_backend_dev_count());
    printf("**************************************************************\nDetected %d devices:\n", nDev);
    for (int i = 0; i < nDev; ++i)
    {
        ggml_backend_dev_t d = ggml_backend_dev_get(i);
        printf("[DEV %d] %s — %s\n", i, ggml_backend_dev_name(d), ggml_backend_dev_description(d));
    }
    fflush(stdout);
    int selected = 0;
    if (nDev > 1)
    {
        printf("Select a device (0-%d): ", nDev - 1);
        fflush(stdout);
        if (scanf("%d", &selected) != 1 || selected < 0 || selected >= nDev) selected = 0;
    }
    ggml_backend_dev_t dev = ggml_backend_dev_get(selected);
    printf("[DEV %d] is Selected.\n**************************************************************\n\n", selected);

    ggml_backend_dev_t devices[2] = { dev, nullptr };
    llama_model_params mparams = llama_model_default_params();
    mparams.devices = devices;
    mparams.n_gpu_layers = 999;
    printf("model: gemma3 %s (%s)\n", args.size.c_str(), args.precision.c_str());
    llama_model* model = llama_model_load_from_file(modelPath.c_str(), mparams);
    if (!model) fail("failed to load model GGUF");
    const llama_vocab* vocab = llama_model_get_vocab(model);

    llama_sampler* chain = llama_sampler_chain_init(llama_sampler_chain_default_params());
    llama_sampler_chain_add(chain, llama_sampler_init_top_k(40));   // vAi TopKSampling default
    llama_sampler_chain_add(chain, llama_sampler_init_dist(seed));
    llama_sampler_seq_config samplerCfg = { 0, chain };

    llama_context_params cparams = llama_context_default_params();
    cparams.n_ctx = nCtx;
    cparams.n_batch = nCtx;
    cparams.n_ubatch = nCtx;
    cparams.n_outputs_max = 1;
    cparams.type_k = cacheType;
    cparams.type_v = cacheType;
    cparams.samplers = &samplerCfg;
    cparams.n_samplers = 1;
    llama_context* ctx = llama_init_from_model(model, cparams);
    if (!ctx) fail("failed to create context");

    mtmd_context* mctx = nullptr;
    mtmd_bitmap* bitmap = nullptr;
    std::vector<uint8_t> imagePixels;

    if (hasVision)
    {
        // At the pinned b10426, mtmd has no device field; it picks its backend
        // from MTMD_BACKEND_DEVICE, so point it at the same device as the model.
#ifdef _WIN32
        _putenv_s("MTMD_BACKEND_DEVICE", ggml_backend_dev_name(dev));
#else
        setenv("MTMD_BACKEND_DEVICE", ggml_backend_dev_name(dev), 1);
#endif
        mtmd_context_params mp = mtmd_context_params_default();
        mp.use_gpu = true;
        mp.warmup = false;
        mctx = mtmd_init_from_file(mmprojPath.c_str(), model, mp);
        if (!mctx || !mtmd_support_vision(mctx)) fail("failed to load vision mmproj GGUF");

        std::vector<uint8_t> imagePixels;

        if(args.imagepath == "") imagePixels = gemma3_bench::makeRandom(gemma3_bench::syntheticImageSize, gemma3_bench::syntheticImageSize, 3);
        else
        {
            int width, height, channels;
            uint8_t* pixels = stbi_load(args.imagepath.c_str(), &width, &height, &channels, 3);
            if (!pixels) fail("failed to load image");
            imagePixels.assign(pixels, pixels + size_t(width) * height * 3);
            stbi_image_free(pixels);
        }

        bitmap = mtmd_bitmap_init(gemma3_bench::syntheticImageSize,
                                  gemma3_bench::syntheticImageSize,
                                  imagePixels.data());
        if (!bitmap) fail("failed to create image bitmap");
    }

    const std::string vqaPrompt = gemma3_bench::vqaPrompt();
    std::vector<llama_token> vqaTokens(vqaPrompt.size() + 16);
    const int nVqa = llama_tokenize(vocab, vqaPrompt.c_str(),
                                    static_cast<int32_t>(vqaPrompt.size()),
                                    vqaTokens.data(), static_cast<int32_t>(vqaTokens.size()),
                                    false, false);
    if (nVqa <= 0) fail("VQA prompt tokenization failed");
    vqaTokens.resize(nVqa);
    for (int n : args.promptLengths)
    {
        if (n > nVqa)
        {
            fprintf(stderr, "error: the VQA prompt has %d tokens, %d requested\n", nVqa, n);
            return 1;
        }
    }
    printHeader(nGen, nReps, nPPReps);
    std::vector<llama_token> sanity;
    int sanityText = 0, sanityTotal = 0;
    for (const int nText : args.promptLengths)
    {
        mtmd_input_chunks* chunks = nullptr;
        std::vector<llama_token> textTokens;
        int nPrompt = 0;
        if (hasVision)
        {
            const std::string question = tokensToString(vocab, vqaTokens, nText);
            const std::string prompt = visionPrompt(mtmd_get_marker(mctx), question);
            chunks = mtmd_input_chunks_init();
            mtmd_input_text inputText = { prompt.c_str(), prompt.size(), false, true };
            const mtmd_bitmap* bitmaps[] = { bitmap };
            if (mtmd_tokenize(mctx, chunks, &inputText, bitmaps, 1))
                fail("multimodal tokenization failed");
            for (size_t i = 0; i < mtmd_input_chunks_size(chunks); ++i)
                nPrompt += static_cast<int>(mtmd_input_chunk_get_n_tokens(mtmd_input_chunks_get(chunks, i)));
        }
        else
        {
            textTokens.assign(vqaTokens.begin(), vqaTokens.begin() + nText);
            nPrompt = nText;
        }
        if (nPrompt + nGen > nCtx) fail("prompt + generation exceeds the context");

        auto prefill = [&]() -> llama_token {
            llama_memory_clear(llama_get_memory(ctx), false);
            if (hasVision)
            {
                llama_pos nPast = 0;
                for (size_t i = 0; i < mtmd_input_chunks_size(chunks); ++i)
                {
                    const mtmd_input_chunk* chunk = mtmd_input_chunks_get(chunks, i);
                    const size_t nChunk = mtmd_input_chunk_get_n_tokens(chunk);
                    if (mtmd_input_chunk_get_type(chunk) == MTMD_INPUT_CHUNK_TYPE_TEXT)
                    {
                        size_t count = 0;
                        const llama_token* text = mtmd_input_chunk_get_tokens_text(chunk, &count);
                        llama_batch batch = llama_batch_get_one(const_cast<llama_token*>(text), static_cast<int32_t>(count));
                        // Prefix chunks seed KV only; compute/sample logits once,
                        // at the final text token, as the vAi graph does.
                        std::vector<int8_t> logits(count, 0);
                        if (i + 1 == mtmd_input_chunks_size(chunks)) logits.back() = 1;
                        batch.logits = logits.data();
                        if (llama_decode(ctx, batch)) fail("multimodal text prefill failed");
                        llama_synchronize(ctx);
                    }
                    else
                    {
                        if (mtmd_encode_chunk(mctx, chunk)) fail("vision encoding failed");
                        const int nEmbd = llama_model_n_embd_inp(model);
                        llama_batch batch = llama_batch_init(static_cast<int32_t>(nChunk), nEmbd, 1);
                        memcpy(batch.embd, mtmd_get_output_embd(mctx), nChunk * nEmbd * sizeof(float));
                        for (size_t j = 0; j < nChunk; ++j)
                        {
                            batch.pos[j] = nPast + static_cast<llama_pos>(j);
                            batch.n_seq_id[j] = 1;
                            batch.seq_id[j][0] = 0;
                            batch.logits[j] = false;
                        }
                        batch.n_tokens = static_cast<int32_t>(nChunk);
                        llama_set_causal_attn(ctx, !mtmd_decode_use_non_causal(mctx, chunk));
                        const int rc = llama_decode(ctx, batch);
                        llama_synchronize(ctx);
                        llama_set_causal_attn(ctx, true);
                        llama_batch_free(batch);
                        if (rc) fail("multimodal image prefill failed");
                    }
                    nPast += mtmd_input_chunk_get_n_pos(chunk);
                }
            }
            else
            {
                llama_batch batch = llama_batch_get_one(textTokens.data(), static_cast<int32_t>(textTokens.size()));
                if (llama_decode(ctx, batch)) fail("text prefill failed");
                llama_synchronize(ctx);
            }
            const llama_token tok = llama_get_sampled_token_ith(ctx, -1);
            if (tok == LLAMA_TOKEN_NULL) fail("backend sampling is inactive");
            return tok;
        };
        auto decodeOne = [&](llama_token tok) -> llama_token {
            llama_batch batch = llama_batch_get_one(&tok, 1);
            if (llama_decode(ctx, batch)) fail("decode failed");
            llama_synchronize(ctx);
            return llama_get_sampled_token_ith(ctx, -1);
        };

        llama_token tok = LLAMA_TOKEN_NULL;
        for (int i = 0; i < nWarmPre; ++i) tok = prefill();
        for (int i = 0; i < nWarmGen; ++i) tok = decodeOne(tok);

        // Prefill, timed on its own: vision encoder + LLM prefill per sample.
        std::vector<double> ppTps;
        for (int rep = 0; rep < nPPReps; ++rep)
        {
            const double t0 = nowSec();
            prefill();
            ppTps.push_back(nPrompt / (nowSec() - t0));
        }

        // Decode, timed on its own: the prefill that seeds each run is not counted.
        std::vector<double> tgTps;
        std::vector<llama_token> generated;
        for (int rep = 0; rep < nReps; ++rep)
        {
            tok = prefill();
            const bool keep = args.showText && rep == 0;
            if (keep) generated.push_back(tok);
            const double t1 = nowSec();
            for (int gen = 0; gen < nGen; ++gen)
            {
                tok = decodeOne(tok);
                if (keep) generated.push_back(tok);
            }
            tgTps.push_back(nGen / (nowSec() - t1));
        }

        printRow("llama.cpp", nText, nPrompt, stats(ppTps), stats(tgTps));
        sanity = std::move(generated);
        sanityText = nText;
        sanityTotal = nPrompt;
        if (chunks) mtmd_input_chunks_free(chunks);
    }
    if (args.showText)
    {
        printf("\n=== Generated Text (text=%d, total=%d, rep 0) ===\n",
               sanityText, sanityTotal);
        for (llama_token t : sanity)
        {
            char buf[256];
            const int n = llama_token_to_piece(vocab, t, buf, sizeof(buf), 0, false);
            if (n > 0) printf("%.*s", n, buf);
        }
        printf("\n");
    }

    if (bitmap) mtmd_bitmap_free(bitmap);
    if (mctx) mtmd_free(mctx);
    llama_free(ctx);
    llama_sampler_free(chain);
    llama_model_free(model);
    llama_backend_free();
    return 0;
}
