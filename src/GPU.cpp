#include "GPU.h"

namespace gpu {
    std::string setupGPUBackend(cv::dnn::Net& net) {
    // FIX OPENCL
    // if (cv::ocl::haveOpenCL()) {
    //     cv::ocl::setUseOpenCL(true);
    //
    //     try {
    //         net.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
    //         net.setPreferableTarget(cv::dnn::DNN_TARGET_OPENCL);
    //
    //         return "OpenCL";
    //     } catch (const cv::Exception& e) {
    //     }
    // } else {
    // }
    //
    // try {
    //     net.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
    //     net.setPreferableTarget(cv::dnn::DNN_TARGET_OPENCL_FP16);
    //
    //     return "OpenCL_FP16";
    // } catch (const cv::Exception& e) {
    // }

    net.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
    net.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);
    return "CPU";
}
} // gpu