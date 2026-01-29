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
    private:


    };
    void detectEdgesBumper(
        cv::Mat& blankFrame,
        cv::Mat& frame,
        const std::vector<Detection>& detections
        );
} // det

#endif //BUMPERDETECTION_EDGEDETECTION_H