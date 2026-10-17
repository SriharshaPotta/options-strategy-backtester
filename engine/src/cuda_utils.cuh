#pragma once

#include <cuda_runtime.h>

#include <cstddef>
#include <stdexcept>
#include <string>

#define CUDA_CHECK(expr)                                                                              \
    do {                                                                                              \
        const cudaError_t err__ = (expr);                                                             \
        if (err__ != cudaSuccess) {                                                                   \
            throw std::runtime_error(std::string("CUDA error: ") + cudaGetErrorString(err__) + " at " + \
                                     __FILE__ + ":" + std::to_string(__LINE__) + " (" #expr ")");     \
        }                                                                                             \
    } while (0)

namespace mc::gpu {

// Minimal RAII owner for a device allocation. Grows on demand, never shrinks.
template <typename T>
class DeviceBuffer {
public:
    DeviceBuffer() = default;
    explicit DeviceBuffer(std::size_t n) { reserve(n); }
    ~DeviceBuffer() { release(); }

    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;
    DeviceBuffer(DeviceBuffer&& o) noexcept : ptr_(o.ptr_), size_(o.size_) { o.ptr_ = nullptr; o.size_ = 0; }

    void reserve(std::size_t n) {
        if (n <= size_) return;
        release();
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&ptr_), n * sizeof(T)));
        size_ = n;
    }

    T* get() const { return ptr_; }
    std::size_t size() const { return size_; }

private:
    void release() {
        if (ptr_) cudaFree(ptr_);
        ptr_ = nullptr;
        size_ = 0;
    }

    T* ptr_ = nullptr;
    std::size_t size_ = 0;
};

}  // namespace mc::gpu
