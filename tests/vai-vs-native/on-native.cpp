// Raw-Vulkan side of test-vai-vs-native: the ggml-style-matmul kernel driven
// without vAi — instance, device, descriptor set, pipeline and dispatch are
// all written out by hand (ported from the old vk-matmul-bench harness).
//
// The device is created with only what the kernel needs: shaderFloat16 for the
// fp16 staging/accumulation, and robustBufferAccess because upstream leaves
// the row-tail loads unmasked.

#include "comparison-native.h"

#include <vulkan/vulkan.h>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

#ifndef GGML_STYLE_MATMUL_SPV
#error "GGML_STYLE_MATMUL_SPV must point at the compiled ggml-style-matmul.comp.spv"
#endif

#define CHECK(x)                                                                     \
    do {                                                                             \
        VkResult err_ = (x);                                                         \
        if (err_ != VK_SUCCESS) {                                                    \
            fprintf(stderr, "vulkan error %d at %s:%d\n", err_, __FILE__, __LINE__); \
            exit(1);                                                                 \
        }                                                                            \
    } while (0)

namespace {

// ggml-vulkan.cpp's vk_mat_mat_push_constants, verbatim (17 u32)
struct PushConstants {
    uint32_t M, N, K;
    uint32_t stride_a, stride_b, stride_d;
    uint32_t batch_stride_a, batch_stride_b, batch_stride_d;
    uint32_t base_work_group_z, num_batches;
    uint32_t k_split;
    uint32_t ne02, ne12, broadcast2, broadcast3;
    uint32_t padded_N;
};

struct Ctx {
    VkInstance            instance     = VK_NULL_HANDLE;
    VkPhysicalDevice      phys         = VK_NULL_HANDLE;
    VkDevice              device       = VK_NULL_HANDLE;
    VkQueue               queue        = VK_NULL_HANDLE;
    uint32_t              qfam         = 0;
    float                 tsPeriod     = 0.0f;   // ns per tick
    uint32_t              subgroupSize = 32;
    VkPhysicalDeviceMemoryProperties memprops{};
    VkCommandPool         cmdpool      = VK_NULL_HANDLE;
    VkDescriptorSetLayout dsl          = VK_NULL_HANDLE;
    VkPipelineLayout      playout      = VK_NULL_HANDLE;
    VkQueryPool           qpool        = VK_NULL_HANDLE;
    VkShaderModule        shader       = VK_NULL_HANDLE;
    VkPipeline            pipes[3]     = {};      // one per warptile class, built on demand
};

struct Buf {
    VkBuffer       buf = VK_NULL_HANDLE;
    VkDeviceMemory mem = VK_NULL_HANDLE;
};

Ctx g;

uint32_t findMemType(uint32_t typeBits, VkMemoryPropertyFlags want)
{
    for (uint32_t i = 0; i < g.memprops.memoryTypeCount; i++)
        if ((typeBits & (1u << i)) && (g.memprops.memoryTypes[i].propertyFlags & want) == want)
            return i;
    fprintf(stderr, "no suitable memory type\n");
    exit(1);
}

Buf createBuf(VkDeviceSize size, VkBufferUsageFlags usage, VkMemoryPropertyFlags memflags)
{
    Buf b;
    VkBufferCreateInfo bi{ VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO };
    bi.size = size;
    bi.usage = usage;
    bi.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    CHECK(vkCreateBuffer(g.device, &bi, nullptr, &b.buf));

    VkMemoryRequirements req;
    vkGetBufferMemoryRequirements(g.device, b.buf, &req);
    VkMemoryAllocateInfo ai{ VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO };
    ai.allocationSize = req.size;
    ai.memoryTypeIndex = findMemType(req.memoryTypeBits, memflags);
    CHECK(vkAllocateMemory(g.device, &ai, nullptr, &b.mem));
    CHECK(vkBindBufferMemory(g.device, b.buf, b.mem, 0));
    return b;
}

void destroyBuf(Buf& b)
{
    if (b.buf) vkDestroyBuffer(g.device, b.buf, nullptr);
    if (b.mem) vkFreeMemory(g.device, b.mem, nullptr);
    b = {};
}

VkCommandBuffer beginOnce()
{
    VkCommandBufferAllocateInfo ai{ VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO };
    ai.commandPool = g.cmdpool;
    ai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    ai.commandBufferCount = 1;
    VkCommandBuffer cb;
    CHECK(vkAllocateCommandBuffers(g.device, &ai, &cb));

    VkCommandBufferBeginInfo bi{ VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO };
    bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    CHECK(vkBeginCommandBuffer(cb, &bi));
    return cb;
}

void submitWait(VkCommandBuffer cb)
{
    CHECK(vkEndCommandBuffer(cb));
    VkSubmitInfo si{ VK_STRUCTURE_TYPE_SUBMIT_INFO };
    si.commandBufferCount = 1;
    si.pCommandBuffers = &cb;
    CHECK(vkQueueSubmit(g.queue, 1, &si, VK_NULL_HANDLE));
    CHECK(vkQueueWaitIdle(g.queue));
    vkFreeCommandBuffers(g.device, g.cmdpool, 1, &cb);
}

void upload(Buf& dst, const void* data, VkDeviceSize size)
{
    Buf staging = createBuf(size, VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    void* p;
    CHECK(vkMapMemory(g.device, staging.mem, 0, size, 0, &p));
    memcpy(p, data, size);
    vkUnmapMemory(g.device, staging.mem);

    VkCommandBuffer cb = beginOnce();
    VkBufferCopy region{ 0, 0, size };
    vkCmdCopyBuffer(cb, staging.buf, dst.buf, 1, &region);
    submitWait(cb);
    destroyBuf(staging);
}

VkShaderModule loadShader(const std::string& path)
{
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f) {
        fprintf(stderr, "cannot open %s (build the shaders first)\n", path.c_str());
        exit(1);
    }
    size_t size = (size_t) f.tellg();
    f.seekg(0);
    std::vector<char> code(size);
    f.read(code.data(), size);

    VkShaderModuleCreateInfo smci{ VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO };
    smci.codeSize = size;
    smci.pCode = (const uint32_t*) code.data();
    VkShaderModule mod;
    CHECK(vkCreateShaderModule(g.device, &smci, nullptr, &mod));
    return mod;
}

// ggml scalar warptile, spec ids 0..10:
// { BLOCK_SIZE, BM, BN, BK (ignored for f32), WM, WN, WMITER, TM, TN, TK, WARP }
struct Tile { char name; uint32_t v[11]; };

Tile makeTile(char cls, uint32_t sg)
{
    uint32_t sg8  = std::max(sg, 8u);
    uint32_t sg32 = std::max(sg, 32u);
    uint32_t swm  = (sg == 8) ? 8 : 32;   // s_warptile_wm
    switch (cls) {
        case 'l': return { 'l', { 128,  128, 128, 16, sg8 * 2, 64, 2, 4, 4, 1, sg8 } };
        case 'm': return { 'm', { 128,   64,  64, 16, sg8,     32, 2, 4, 2, 1, sg8 } };
        default:  return { 's', { sg32,  32,  32, 16, swm,     32, 2, 2, 2, 1, sg8 } };
    }
}

// ggml_vk_guess_matmul_pipeline's S/M/L thresholds (ggml M = vai N)
char pickClass(uint32_t ggmlM, uint32_t ggmlN)
{
    if (ggmlM <= 32 || ggmlN <= 32) return 's';
    if (ggmlM <= 64 || ggmlN <= 64) return 'm';
    return 'l';
}

VkPipeline pipelineFor(const Tile& t)
{
    const int slot = (t.name == 's') ? 0 : (t.name == 'm') ? 1 : 2;
    if (g.pipes[slot])
        return g.pipes[slot];

    VkSpecializationMapEntry entries[11];
    for (uint32_t i = 0; i < 11; i++) entries[i] = { i, i * 4, 4 };
    VkSpecializationInfo spec{ 11, entries, sizeof(t.v), t.v };

    VkComputePipelineCreateInfo cpci{ VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO };
    cpci.stage.sType  = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    cpci.stage.stage  = VK_SHADER_STAGE_COMPUTE_BIT;
    cpci.stage.module = g.shader;
    cpci.stage.pName  = "main";
    cpci.stage.pSpecializationInfo = &spec;
    cpci.layout = g.playout;
    CHECK(vkCreateComputePipelines(g.device, VK_NULL_HANDLE, 1, &cpci, nullptr, &g.pipes[slot]));
    return g.pipes[slot];
}

} // namespace


void nativeInit(int gpuIndex)
{
    VkApplicationInfo app{ VK_STRUCTURE_TYPE_APPLICATION_INFO };
    app.pApplicationName = "test-vai-vs-native";
    app.apiVersion = VK_API_VERSION_1_3;
    VkInstanceCreateInfo ici{ VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO };
    ici.pApplicationInfo = &app;
    CHECK(vkCreateInstance(&ici, nullptr, &g.instance));

    uint32_t ndev = 0;
    CHECK(vkEnumeratePhysicalDevices(g.instance, &ndev, nullptr));
    if (ndev == 0) { fprintf(stderr, "no vulkan device\n"); exit(1); }
    std::vector<VkPhysicalDevice> devs(ndev);
    CHECK(vkEnumeratePhysicalDevices(g.instance, &ndev, devs.data()));

    // Explicit index, else the first discrete GPU, else device 0.
    int pick = gpuIndex;
    if (pick < 0) {
        pick = 0;
        for (uint32_t i = 0; i < ndev; i++) {
            VkPhysicalDeviceProperties p;
            vkGetPhysicalDeviceProperties(devs[i], &p);
            if (p.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU) { pick = (int) i; break; }
        }
    }
    g.phys = devs[pick];

    VkPhysicalDeviceSubgroupProperties sgp{ VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_PROPERTIES };
    VkPhysicalDeviceProperties2 props2{ VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2 };
    props2.pNext = &sgp;
    vkGetPhysicalDeviceProperties2(g.phys, &props2);
    g.tsPeriod     = props2.properties.limits.timestampPeriod;
    g.subgroupSize = sgp.subgroupSize;

    printf("[native] device %s, subgroupSize %u\n", props2.properties.deviceName, g.subgroupSize);

    uint32_t nqf = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(g.phys, &nqf, nullptr);
    std::vector<VkQueueFamilyProperties> qfs(nqf);
    vkGetPhysicalDeviceQueueFamilyProperties(g.phys, &nqf, qfs.data());
    g.qfam = UINT32_MAX;
    for (uint32_t i = 0; i < nqf; i++)
        if (qfs[i].queueFlags & VK_QUEUE_COMPUTE_BIT) { g.qfam = i; break; }
    if (g.qfam == UINT32_MAX) { fprintf(stderr, "no compute queue\n"); exit(1); }
    if (qfs[g.qfam].timestampValidBits == 0 || g.tsPeriod <= 0.0f) {
        fprintf(stderr, "queue has no usable timestamps — cannot benchmark\n");
        exit(1);
    }

    VkPhysicalDeviceVulkan12Features f12{ VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES };
    VkPhysicalDeviceFeatures2 f2{ VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2 };
    f2.pNext = &f12;
    vkGetPhysicalDeviceFeatures2(g.phys, &f2);
    if (!f12.shaderFloat16)              { fprintf(stderr, "device lacks shaderFloat16\n"); exit(1); }
    if (!f2.features.robustBufferAccess) { fprintf(stderr, "device lacks robustBufferAccess\n"); exit(1); }

    VkPhysicalDeviceVulkan12Features enable12{ VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES };
    enable12.shaderFloat16 = VK_TRUE;
    VkPhysicalDeviceFeatures2 enable2{ VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2 };
    enable2.pNext = &enable12;
    enable2.features.robustBufferAccess = VK_TRUE;

    float prio = 1.0f;
    VkDeviceQueueCreateInfo qci{ VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO };
    qci.queueFamilyIndex = g.qfam;
    qci.queueCount = 1;
    qci.pQueuePriorities = &prio;
    VkDeviceCreateInfo dci{ VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO };
    dci.queueCreateInfoCount = 1;
    dci.pQueueCreateInfos = &qci;
    dci.pNext = &enable2;
    CHECK(vkCreateDevice(g.phys, &dci, nullptr, &g.device));
    vkGetDeviceQueue(g.device, g.qfam, 0, &g.queue);
    vkGetPhysicalDeviceMemoryProperties(g.phys, &g.memprops);

    VkCommandPoolCreateInfo cpi{ VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO };
    cpi.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    cpi.queueFamilyIndex = g.qfam;
    CHECK(vkCreateCommandPool(g.device, &cpi, nullptr, &g.cmdpool));

    // binding 0 = A (weight), 1 = B (input), 2 = D (output)
    VkDescriptorSetLayoutBinding binds[3]{};
    for (uint32_t i = 0; i < 3; i++) {
        binds[i].binding = i;
        binds[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        binds[i].descriptorCount = 1;
        binds[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    }
    VkDescriptorSetLayoutCreateInfo dslci{ VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO };
    dslci.bindingCount = 3;
    dslci.pBindings = binds;
    CHECK(vkCreateDescriptorSetLayout(g.device, &dslci, nullptr, &g.dsl));

    VkPushConstantRange pcr{ VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(PushConstants) };
    VkPipelineLayoutCreateInfo plci{ VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO };
    plci.setLayoutCount = 1;
    plci.pSetLayouts = &g.dsl;
    plci.pushConstantRangeCount = 1;
    plci.pPushConstantRanges = &pcr;
    CHECK(vkCreatePipelineLayout(g.device, &plci, nullptr, &g.playout));

    VkQueryPoolCreateInfo qpi{ VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO };
    qpi.queryType = VK_QUERY_TYPE_TIMESTAMP;
    qpi.queryCount = 2;
    CHECK(vkCreateQueryPool(g.device, &qpi, nullptr, &g.qpool));

    g.shader = loadShader(GGML_STYLE_MATMUL_SPV);
}

BenchResult nativeRun(const BenchCase& c, int warmup, int reps)
{
    const uint32_t M = c.M, N = c.N, K = c.K;

    // ggml's letters are vai's with M/N swapped, and the kernel's ALIGNED
    // permutation needs whole BK (32) blocks.
    const Tile t = makeTile(pickClass(N, M), g.subgroupSize);
    const uint32_t BM = t.v[1], BN = t.v[2];
    const uint32_t gx = (N + BM - 1) / BM, gy = (M + BN - 1) / BN;

    VkPipeline pipe = pipelineFor(t);

    const VkDeviceSize wSize = (VkDeviceSize) N * K * 4;
    const VkDeviceSize iSize = (VkDeviceSize) M * K * 4;
    const VkDeviceSize oSize = (VkDeviceSize) M * N * 4;
    Buf W = createBuf(wSize, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                      VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    Buf I = createBuf(iSize, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                      VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    Buf O = createBuf(oSize, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                      VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    {
        std::vector<float> hostW((size_t) N * K), hostI((size_t) M * K);
        for (uint32_t n = 0; n < N; n++)
            for (uint32_t k = 0; k < K; k++) hostW[(size_t) n * K + k] = benchInitW(n, k);
        for (uint32_t m = 0; m < M; m++)
            for (uint32_t k = 0; k < K; k++) hostI[(size_t) m * K + k] = benchInitI(m, k);
        upload(W, hostW.data(), wSize);
        upload(I, hostI.data(), iSize);
    }

    VkDescriptorPoolSize psz{ VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 3 };
    VkDescriptorPoolCreateInfo dpci{ VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO };
    dpci.maxSets = 1;
    dpci.poolSizeCount = 1;
    dpci.pPoolSizes = &psz;
    VkDescriptorPool dpool;
    CHECK(vkCreateDescriptorPool(g.device, &dpci, nullptr, &dpool));

    VkDescriptorSetAllocateInfo dsai{ VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO };
    dsai.descriptorPool = dpool;
    dsai.descriptorSetCount = 1;
    dsai.pSetLayouts = &g.dsl;
    VkDescriptorSet dset;
    CHECK(vkAllocateDescriptorSets(g.device, &dsai, &dset));

    VkDescriptorBufferInfo binfo[3] = { { W.buf, 0, wSize }, { I.buf, 0, iSize }, { O.buf, 0, oSize } };
    VkWriteDescriptorSet writes[3]{};
    for (uint32_t i = 0; i < 3; i++) {
        writes[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[i].dstSet = dset;
        writes[i].dstBinding = i;
        writes[i].descriptorCount = 1;
        writes[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        writes[i].pBufferInfo = &binfo[i];
    }
    vkUpdateDescriptorSets(g.device, 3, writes, 0, nullptr);

    // Same values Linear3Node pushes: single 2D GEMM, packed strides, no
    // batching, no k-split.
    PushConstants pc{};
    pc.M = N; pc.N = M; pc.K = K;
    pc.stride_a = K; pc.stride_b = K; pc.stride_d = N;
    pc.batch_stride_a = 0; pc.batch_stride_b = 0; pc.batch_stride_d = 0;
    pc.base_work_group_z = 0;
    pc.num_batches = 1;
    pc.k_split = K;
    pc.ne02 = 1; pc.ne12 = 1; pc.broadcast2 = 1; pc.broadcast3 = 1;
    pc.padded_N = M;

    auto record = [&](VkCommandBuffer cb, bool withTs) {
        vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, pipe);
        vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_COMPUTE, g.playout, 0, 1, &dset, 0, nullptr);
        vkCmdPushConstants(cb, g.playout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);
        if (withTs) {
            vkCmdResetQueryPool(cb, g.qpool, 0, 2);
            vkCmdWriteTimestamp(cb, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, g.qpool, 0);
        }
        vkCmdDispatch(cb, gx, gy, 1);
        if (withTs)
            vkCmdWriteTimestamp(cb, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, g.qpool, 1);
    };

    for (int i = 0; i < warmup; i++) {
        VkCommandBuffer cb = beginOnce();
        record(cb, false);
        submitWait(cb);
    }

    // One matmul per submit; the timestamps bracket the dispatch only, so
    // submit/wait CPU overhead stays out of the total.
    double totalMs = 0.0;
    for (int r = 0; r < reps; r++) {
        VkCommandBuffer cb = beginOnce();
        record(cb, true);
        submitWait(cb);

        uint64_t ts[2];
        CHECK(vkGetQueryPoolResults(g.device, g.qpool, 0, 2, sizeof(ts), ts, 8,
                                    VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WAIT_BIT));
        totalMs += double(ts[1] - ts[0]) * g.tsPeriod / 1e6;
    }

    vkDestroyDescriptorPool(g.device, dpool, nullptr);
    destroyBuf(W);
    destroyBuf(I);
    destroyBuf(O);

    return { totalMs / reps, t.name, gx, gy };
}

void nativeShutdown()
{
    for (VkPipeline p : g.pipes) if (p) vkDestroyPipeline(g.device, p, nullptr);
    if (g.shader)  vkDestroyShaderModule(g.device, g.shader, nullptr);
    if (g.qpool)   vkDestroyQueryPool(g.device, g.qpool, nullptr);
    if (g.playout) vkDestroyPipelineLayout(g.device, g.playout, nullptr);
    if (g.dsl)     vkDestroyDescriptorSetLayout(g.device, g.dsl, nullptr);
    if (g.cmdpool) vkDestroyCommandPool(g.device, g.cmdpool, nullptr);
    if (g.device)  vkDestroyDevice(g.device, nullptr);
    if (g.instance) vkDestroyInstance(g.instance, nullptr);
    g = {};
}
