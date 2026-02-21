#include "GPU.h"
#include <opencv2/core/ocl.hpp>

namespace gpu {
    std::string setupGPUBackend(cv::dnn::Net& net) {

#ifdef HAVE_OPENCV_CUDNN
        try {
            net.setPreferableBackend(cv::dnn::DNN_BACKEND_CUDA);
            net.setPreferableTarget(cv::dnn::DNN_TARGET_CUDA_FP16);
            return "CUDA_FP16";
        } catch (const cv::Exception& e) {
        }
#endif
    net.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
    net.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);
    return "CPU";
}
}