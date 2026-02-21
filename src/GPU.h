#ifndef BUMPERDETECTION_SETUPGPU_H
#define BUMPERDETECTION_SETUPGPU_H
#include <opencv2/opencv.hpp>

std::string setupGPUBackend(cv::dnn::Net &net);

#endif //BUMPERDETECTION_SETUPGPU_H
