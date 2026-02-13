#include "GPU.h"
#include <iostream>

#include <opencv2/core/ocl.hpp>

namespace gpu {
    std::string setupGPUBackend(cv::dnn::Net& net) {
    std::cout << "\n[GPU DETECTION]" << std::endl;
    std::cout << "Testing available backends..." << std::endl;

    // Priority 1: Try CUDA (fastest)
    #ifdef HAVE_OPENCV_CUDNN
    try {
        int cuda_device_count = cv::cuda::getCudaEnabledDeviceCount();
        if (cuda_device_count > 0) {
            std::cout << "\n[CUDA Detection]" << std::endl;
            std::cout << "CUDA devices available: " << cuda_device_count << std::endl;

            cv::cuda::DeviceInfo deviceInfo;
            std::cout << "GPU: " << deviceInfo.name() << std::endl;
            std::cout << "Compute Capability: " << deviceInfo.majorVersion()
                      << "." << deviceInfo.minorVersion() << std::endl;

            std::cout << "\nAttempting CUDA backend..." << std::endl;
            net.setPreferableBackend(cv::dnn::DNN_BACKEND_CUDA);
            net.setPreferableTarget(cv::dnn::DNN_TARGET_CUDA);

            std::cout << "Using CUDA for GPU acceleration" << std::endl;
            return "CUDA";
        }
        std::cout << "No CUDA devices found" << std::endl;
    } catch (const cv::Exception& e) {
        std::cout << "CUDA failed: " << e.what() << std::endl;
    }
    #else
    std::cout << "✗ OpenCV not built with CUDA support" << std::endl;
    std::cout << "  To enable CUDA, rebuild OpenCV with:" << std::endl;
    std::cout << "  - OPENCV_DNN_CUDA=ON" << std::endl;
    std::cout << "  - WITH_CUDA=ON" << std::endl;
    std::cout << "  - WITH_CUDNN=ON" << std::endl;
    #endif

    // Priority 2: Try CUDA FP16 (even faster, slightly less accurate)
    #ifdef HAVE_OPENCV_CUDNN
    try {
        std::cout << "\nAttempting CUDA FP16 backend..." << std::endl;
        net.setPreferableBackend(cv::dnn::DNN_BACKEND_CUDA);
        net.setPreferableTarget(cv::dnn::DNN_TARGET_CUDA_FP16);

        std::cout << "Using CUDA FP16 for GPU acceleration (VERY FAST)" << std::endl;
        return "CUDA_FP16";
    } catch (const cv::Exception& e) {
        std::cout << "CUDA FP16 failed: " << e.what() << std::endl;
    }
    #endif

    // Priority 3: Try OpenCL (cross-platform)
    std::cout << "\nAttempting OpenCL backend..." << std::endl;
    if (cv::ocl::haveOpenCL()) {
        std::cout << "OpenCL runtime: AVAILABLE" << std::endl;
        cv::ocl::setUseOpenCL(true);

        cv::ocl::Device device = cv::ocl::Device::getDefault();
        std::cout << "OpenCL Device: " << device.name() << std::endl;
        std::cout << "OpenCL Vendor: " << device.vendorName() << std::endl;

        try {
            net.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
            net.setPreferableTarget(cv::dnn::DNN_TARGET_OPENCL);

            std::cout << "Using OpenCL for GPU acceleration" << std::endl;
            return "OpenCL";
        } catch (const cv::Exception& e) {
            std::cout << "OpenCL failed: " << e.what() << std::endl;
        }
    } else {
        std::cout << "OpenCL not available" << std::endl;
    }

    // Priority 4: Try OpenCL FP16
    try {
        std::cout << "\nAttempting OpenCL FP16 backend..." << std::endl;
        net.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
        net.setPreferableTarget(cv::dnn::DNN_TARGET_OPENCL_FP16);

        std::cout << "Using OpenCL FP16 for GPU acceleration" << std::endl;
        return "OpenCL_FP16";
    } catch (const cv::Exception& e) {
        std::cout << "OpenCL FP16 failed: " << e.what() << std::endl;
    }

    // Fallback: CPU
    std::cout << "\nFalling back to CPU (no GPU acceleration)" << std::endl;
    net.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
    net.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);
    return "CPU";
}
} // gpu