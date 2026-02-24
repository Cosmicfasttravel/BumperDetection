#include "GPU.h"

#include <opencv2/core/ocl.hpp>
using namespace std;
namespace gpu {
    std::string setupGPUBackend(cv::dnn::Net &net) {
        net.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
        net.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);
        return "CPU";
    }
} // gpu
