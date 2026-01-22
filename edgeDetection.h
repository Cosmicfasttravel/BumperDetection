#ifndef BUMPERDETECTION_EDGEDETECTION_H
#define BUMPERDETECTION_EDGEDETECTION_H
#include <opencv2/core/mat.hpp>

struct Detection {
    int class_id;
    float confidence;
    cv::Rect bounding_box;
    std::string class_name;
};

namespace det {
    class edgeDetection {
        public:


    };
    void detectEdgesBumper(
        cv::Mat& blankFrame,
        cv::Mat& frame,
        const std::vector<Detection>& detections,
        double focal_length = 833.0,
        double bumper_width_cm = 30.0
        );
} // det

#endif //BUMPERDETECTION_EDGEDETECTION_H