#include "process_detection.h"
#include "core/detection/structure/detection_data.h"

namespace process_detection {
    std::vector<Detection> ProcessYoloOutput(
    const std::vector<cv::Mat> &outputs,
    int img_width,
    int img_height,
    int input_width,
    int input_height,
    float conf_threshold,
    float nms_threshold)
{
    std::vector<Detection> detections;
    std::vector<int> class_ids;
    std::vector<float> confidences;
    std::vector<cv::Rect> boxes;

    float scale_x = static_cast<float>(img_width) / static_cast<float>(input_width);
    float scale_y = static_cast<float>(img_height) / static_cast<float>(input_height);

    cv::Mat output_data = outputs[0];

    int num_detections;
    int num_values_per_detection;

    if (output_data.dims == 3)
    {
        int dim1 = output_data.size[1];
        int dim2 = output_data.size[2];

        if (dim1 < 100 && dim2 > 1000)
        {
            output_data = output_data.reshape(1, dim1);
            cv::transpose(output_data, output_data);
            num_detections = dim2;
            num_values_per_detection = dim1;
        }
        else
        {
            output_data = output_data.reshape(1, dim1);
            num_detections = dim1;
            num_values_per_detection = dim2;
        }
    }
    else if (output_data.dims == 2)
    {
        num_detections = output_data.rows;
        num_values_per_detection = output_data.cols;
    }
    else
    {
        return detections;
    }

    bool is_single_class = (num_values_per_detection == 5);
    int num_classes = is_single_class ? 1 : (num_values_per_detection - 4);

    for (int i = 0; i < num_detections; ++i)
    {
        auto *data = output_data.ptr<float>(i);

        float x_center = data[0];
        float y_center = data[1];
        float width = data[2];
        float height = data[3];

        float max_score;
        int max_class_id = 0;

        if (is_single_class)
        {
            max_score = data[4];
        }
        else
        {
            float *class_scores = data + 4;
            max_score = -1.0f;
            max_class_id = -1;

            for (int c = 0; c < num_classes; c++)
            {
                if (class_scores[c] > max_score)
                {
                    max_score = class_scores[c];
                    max_class_id = c;
                }
            }
        }

        if (max_score < conf_threshold)
        {
            continue;
        }

        bool is_normalized = (x_center <= 1.0f && y_center <= 1.0f &&
                              width <= 1.0f && height <= 1.0f);

        if (is_normalized)
        {
            x_center *= static_cast<float>(img_width);
            y_center *= static_cast<float>(img_height);
            width *= static_cast<float>(img_width);
            height *= static_cast<float>(img_height);
        }
        else
        {
            x_center *= scale_x;
            y_center *= scale_y;
            width *= scale_x;
            height *= scale_y;
        }

        int x = static_cast<int>(x_center - width / 2.0f);
        int y = static_cast<int>(y_center - height / 2.0f);
        int w = static_cast<int>(width);
        int h = static_cast<int>(height);

        x = std::max(0, std::min(x, img_width - 1));
        y = std::max(0, std::min(y, img_height - 1));
        w = std::max(1, std::min(w, img_width - x));
        h = std::max(1, std::min(h, img_height - y));

        class_ids.emplace_back(max_class_id);
        confidences.emplace_back(max_score);
        boxes.emplace_back(x, y, w, h);
    }

    std::vector<int> nms_result;
    cv::dnn::NMSBoxesBatched(
        boxes,
        confidences,
        class_ids,
        conf_threshold,
        nms_threshold,
        nms_result);

    for (int idx : nms_result)
    {
        Detection result;
        result.confidence = confidences[idx];
        result.boundingBox = boxes[idx];
        if (result.boundingBox.width >= 450 || result.boundingBox.width <= 50)
        {
            continue;
        }

        detections.emplace_back(result);
    }

    return detections;
}
}