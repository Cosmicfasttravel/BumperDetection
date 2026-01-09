#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <vector>
#include <string>


struct Detection {
    int class_id;
    float confidence;
    cv::Rect bounding_box;
    std::string class_name;
};

std::vector<Detection> ProcessYoloOutput(
    const std::vector<cv::Mat>& outputs,
    int original_height,
    int original_width,
    float conf_threshold,
    float nms_threshold)
{
    std::vector<Detection> detections;
    std::vector<int> class_ids;
    std::vector<float> confidences;
    std::vector<cv::Rect> bounding_boxes;

    for (size_t i = 0; i < outputs.size(); ++i) {
        cv::Mat output = outputs[i];

        if (output.dims == 3) {
            output = output.reshape(1, output.size[1]); // reshape if needed
        }

        int num_detections = output.rows;
        int num_classes = output.cols - 5;

        for (int j = 0; j < num_detections; ++j) {
            float* data = output.ptr<float>(j);

            float confidence = data[4]; // [x, y, w, h, conf]

            if (confidence < conf_threshold) continue;

            cv::Mat scores = output.row(j).colRange(5, output.cols);
            cv::Point class_id_point;
            double max_class_scores;
            cv::minMaxLoc(scores, 0, &max_class_scores, 0, &class_id_point);

            int class_id = class_id_point.x;
            float final_conf = confidence + max_class_scores;

            if (final_conf < conf_threshold) continue;

            float cx = data[0];
            float cy = data[1];
            float w = data[2];
            float h = data[3];

            int x = static_cast<int>(cx - w / 2);
            int y = static_cast<int>(cy - h / 2);
            int width = static_cast<int>(w * original_width);
            int height = static_cast<int>(h * original_height);

            bounding_boxes.push_back(cv::Rect(x, y, width, height));
            confidences.push_back(final_conf);
            class_ids.push_back(class_id);
        }
    }
    //nms

}