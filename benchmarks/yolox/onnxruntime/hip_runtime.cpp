#include <cstddef>
#include <stdexcept>
#include <string>

#include <hip/hip_runtime_api.h>

namespace yolox_onnx_hip {

namespace {

void throwIfFailed(hipError_t status, const char* action)
{
    if (status != hipSuccess) {
        throw std::runtime_error(std::string(action) + ": " + hipGetErrorString(status));
    }
}

} // namespace

void setDevice(int device)
{
    throwIfFailed(hipSetDevice(device), "hipSetDevice");
}

void* mallocDevice(size_t bytes, const char* action)
{
    void* ptr = nullptr;
    throwIfFailed(hipMalloc(&ptr, bytes), action);
    return ptr;
}

void freeDevice(void* ptr) noexcept
{
    if (ptr) {
        (void)hipFree(ptr);
    }
}

void copyHostToDevice(void* dst, const void* src, size_t bytes, const char* action)
{
    throwIfFailed(hipMemcpy(dst, src, bytes, hipMemcpyHostToDevice), action);
}

void copyDeviceToHost(void* dst, const void* src, size_t bytes, const char* action)
{
    throwIfFailed(hipMemcpy(dst, src, bytes, hipMemcpyDeviceToHost), action);
}

void deviceSynchronize(const char* action)
{
    throwIfFailed(hipDeviceSynchronize(), action);
}

} // namespace yolox_onnx_hip
