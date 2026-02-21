#include "GPU.h"
#include <opencv2/core/ocl.hpp>

namespace gpu {
    std::string setupGPUBackend(cv::dnn::Net& net) {

    try {
        int cuda_device_count = cv::cuda::getCudaEnabledDeviceCount();
        if (cuda_device_count > 0) {
            net.setPreferableBackend(cv::dnn::DNN_BACKEND_CUDA);
            net.setPreferableTarget(cv::dnn::DNN_TARGET_CUDA);

            return "CUDA";
        }
    } catch (const cv::Exception& e) {
    }

    try {
        net.setPreferableBackend(cv::dnn::DNN_BACKEND_CUDA);
        net.setPreferableTarget(cv::dnn::DNN_TARGET_CUDA_FP16);

        return "CUDA_FP16";
    } catch (const cv::Exception& e) {
    }

        return "";
}
} // gpu