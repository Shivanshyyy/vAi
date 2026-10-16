// Tokenizer against the reference implementation: a fixed set of inputs and the
// ids HuggingFace tokenizers 0.23.1 produced for each tokenizer.json under data/
// (0.22.2 for the paligemma set).
// Encoding has to match id for id, and decoding has to give the input back.
//
//   test-tokenizer
//
// The checkpoints are not in the repository, so a tokenizer.json that is not on
// disk is reported and skipped rather than failing the run.

#include "vai/utils/tokenizer.h"
#include <cstdio>
#include <fstream>
#include <string>
#include <vector>


// Whitespace runs, punctuation runs, digits glued to letters, three non-Latin
// scripts, astral-plane emoji, a trailing space, and the empty string. Between
// them these cover what tells the pre-tokenizers apart: a Split pattern the
// normalizer has already rewritten leaves nothing to match, and every model here
// spells its whitespace differently.
const char* kInputs[] = {
    "Hello world",
    "In recent decades, artificial intelligence has progressed",
    "  leading   spaces and\ttabs\nnewlines\n\n",
    "punctuation!!! ??? mixed123numbers456 CAPS lower",
    "unicode: \xed\x95\x9c\xea\xb5\xad\xec\x96\xb4 \xe6\x97\xa5\xe6\x9c\xac\xe8\xaa\x9e"
    " \xd0\xa0\xd1\x83\xd1\x81\xd1\x81\xd0\xba\xd0\xb8\xd0\xb9",
    "emoji \xf0\x9f\x9a\x80\xf0\x9f\x8e\x89 and symbols \xc2\xa9\xc2\xae\xe2\x84\xa2",
    "trailing space ",
    "",
};

constexpr size_t kNumInputs = sizeof(kInputs) / sizeof(kInputs[0]);


// One unbroken run of letters. The pre-tokenizer's \p{L}+ claims all of it, so
// the whole run arrives at a single BPE call, and a repetition that recurses once
// per code point puts the run length on the stack. The reference collapses it to
// one token repeated, which is why the expectation is a count and not a list.
constexpr size_t kLongRunLength = 5000;

struct Expected
{
    const char*                   tokenizerJson;
    std::vector<std::vector<int>> ids;      // one entry per kInputs, in order
    int    longRunId;                       // the token the run comes out as
    size_t longRunCount;
};

// ByteLevel BPE without a Split stage (gpt2), ByteLevel with one (llama3), and
// SentencePiece with a normalizer the decoder has to undo (gemma3). PaliGemma's
// Gemma 1 vocabulary is the same family with the byte <0x09> spelled as a raw
// "\t" (id 226), which the tab in case 2 has to come out as.
const std::vector<Expected> kExpected = {
    { VAI_DATA_DIR "/gpt2/tokenizer.json", {
        { 15496, 995 },
        { 818, 2274, 4647, 11, 11666, 4430, 468, 28775 },
        { 220, 3756, 220, 220, 9029, 290, 197, 8658, 82, 198, 3605, 6615, 628 },
        { 79, 16260, 2288, 10185, 34913, 7668, 10163, 77, 17024, 29228, 20176, 50, 2793 },
        {
            46903, 1098, 25, 220, 47991, 250, 166, 113, 255, 168, 244, 112, 10545, 245, 98,
            17312, 105, 45739, 252, 12466, 254, 35072, 21727, 21727, 31583, 18849, 140, 117,
        },
        { 368, 31370, 12520, 248, 222, 8582, 236, 231, 290, 14354, 10673, 7461, 8151 },
        { 9535, 4386, 2272, 220 },
        {  },
    }, 24794, 1250 },
    { VAI_DATA_DIR "/llama3/tokenizer.json", {
        { 9906, 1917 },
        { 644, 3293, 11026, 11, 21075, 11478, 706, 62916 },
        { 220, 6522, 256, 12908, 323, 3324, 3518, 198, 943, 8128, 271 },
        { 79, 73399, 12340, 52417, 9709, 4513, 38478, 10961, 27193, 50, 4827 },
        { 25535, 25, 104008, 32179, 105180, 102158, 121258, 101591 },
        { 38623, 11410, 248, 222, 9468, 236, 231, 323, 18210, 7388, 12175, 16500 },
        { 376, 14612, 3634, 220 },
        {  },
    }, 70540, 625 },
    { VAI_DATA_DIR "/gemma3/tokenizer.json", {
        { 9259, 1902 },
        { 902, 4803, 13013, 236764, 16477, 14020, 815, 65426 },
        { 138, 26016, 139, 35220, 532, 255968, 39218, 107, 208697, 108 },
        {
            196235, 11145, 81001, 9726, 236770, 236778, 236800, 34488, 236812, 236810, 236825,
            182027, 3718,
        },
        { 70926, 236787, 50413, 237430, 33375, 238582, 214089 },
        { 67906, 236743, 242015, 242061, 532, 16271, 17761, 238216, 238590 },
        { 136697, 2557, 236743 },
        {  },
    }, 128820, 625 },
    { VAI_DATA_DIR "/paligemma/tokenizer.json", {
        { 4521, 2134 },
        { 886, 6077, 17481, 235269, 18225, 17273, 919, 77885 },
        { 139, 26650, 140, 46633, 578, 226, 31973, 108, 1404, 5448, 109 },
        {
            207854, 4762, 35232, 10616, 235274, 235284, 235304, 28359, 235310, 235308, 235318,
            175801, 4581,
        },
        { 56607, 235292, 114490, 236770, 26867, 236333, 203914 },
        { 62400, 167340, 239548, 578, 22013, 11532, 236205, 236649 },
        { 100504, 3641, 235248 },
        {  },
    }, 84377, 625 },
};


void printIds(const char* label, const std::vector<int>& ids)
{
    printf("      %s:", label);
    for (int id : ids)
        printf(" %d", id);
    printf("\n");
}


int main()
{
    int failed = 0, skipped = 0, checked = 0;

    for (const Expected& expected : kExpected)
    {
        printf("%s\n", expected.tokenizerJson);

        // Only a missing file is a skip; anything else the loader rejects is a
        // real failure and has to surface.
        if (!std::ifstream(expected.tokenizerJson).is_open())
        {
            printf("  [skip] not on disk\n\n");
            ++skipped;
            continue;
        }

        Tokenizer tokenizer(expected.tokenizerJson);

        for (size_t i = 0; i < kNumInputs; ++i)
        {
            const std::string   input = kInputs[i];
            std::vector<int>    ids   = tokenizer.encode(input, /*addSpecial*/ false);
            std::string         text  = tokenizer.decode(ids);

            bool encodeOk = ids == expected.ids[i];
            bool decodeOk = text == input;
            failed += !encodeOk + !decodeOk;
            ++checked;

            printf("  [%s] case %zu  %zu ids\n", encodeOk && decodeOk ? "ok  " : "FAIL", i, ids.size());
            if (!encodeOk)
            {
                printIds("want", expected.ids[i]);
                printIds("got ", ids);
            }
            if (!decodeOk)
                printf("      decode: \"%s\"\n", text.c_str());
        }

        {
            const std::string   run(kLongRunLength, 'a');
            std::vector<int>    ids  = tokenizer.encode(run, /*addSpecial*/ false);
            std::string         text = tokenizer.decode(ids);

            bool countOk = ids.size() == expected.longRunCount;
            bool everyOk = !ids.empty();
            for (int id : ids)
                everyOk = everyOk && id == expected.longRunId;
            bool ok = countOk && everyOk && text == run;
            failed += !ok;
            ++checked;

            printf("  [%s] run of %zu letters  %zu ids\n",
                   ok ? "ok  " : "FAIL", kLongRunLength, ids.size());
            if (!ok)
                printf("      want %zu x %d, decode round-trip %s\n",
                       expected.longRunCount, expected.longRunId,
                       text == run ? "ok" : "broken");
        }
        printf("\n");
    }

    printf("%d checks, %d failed, %d tokenizers skipped\n", checked, failed, skipped);
    return failed ? 1 : 0;
}
