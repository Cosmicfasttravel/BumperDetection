#ifndef BUMPERDETECTION_SETUPGPU_H
#define BUMPERDETECTION_SETUPGPU_H
#include <opencv2/opencv.hpp>

namespace gpu {
    std::string setupGPUBackend(cv::dnn::Net& net);
} // gpu

#endif //BUMPERDETECTION_SETUPGPU_H