#ifndef BUMPERDETECTION_BUILT_BACKEND_H
#define BUMPERDETECTION_BUILT_BACKEND_H

namespace build_info {
    enum class Backend {
        CPU,
        NVIDIA,
        RKNN
    };

    inline constexpr Backend backend =
#ifdef TARGET_NVIDIA_GPU
            Backend::NVIDIA;
#elif defined(TARGET_RKNN_NPU)
            Backend::RKNN;
#else
            Backend::CPU;
#endif

    inline constexpr bool is_cpu = (backend == Backend::CPU);
    inline constexpr bool is_nvidia = (backend == Backend::NVIDIA);
    inline constexpr bool is_rknn = (backend == Backend::RKNN);
}

#endif //BUMPERDETECTION_BUILT_BACKEND_H
