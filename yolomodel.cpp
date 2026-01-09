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
    int img_width,
    int img_height,
    int input_width,   // Add this parameter
    int input_height,  // Add this parameter
    float conf_threshold,
    float nms_threshold
) {
    std::vector<Detection> detections;
    std::vector<int> class_ids;
    std::vector<float> confidences;
    std::vector<cv::Rect> boxes;

    // Calculate scaling factors
    float scale_x = static_cast<float>(img_width) / input_width;
    float scale_y = static_cast<float>(img_height) / input_height;

    std::cout << "\n[DEBUG] Image size: " << img_width << "x" << img_height << std::endl;
    std::cout << "[DEBUG] Input size: " << input_width << "x" << input_height << std::endl;
    std::cout << "[DEBUG] Scale factors: x=" << scale_x << ", y=" << scale_y << std::endl;
    std::cout << "[DEBUG] Number of output blobs: " << outputs.size() << std::endl;

    cv::Mat output_data = outputs[0];

    std::cout << "[DEBUG] Output dimensions: " << output_data.dims << std::endl;
    std::cout << "[DEBUG] Output shape: ";
    for (int i = 0; i < output_data.dims; i++) {
        std::cout << output_data.size[i] << " ";
    }
    std::cout << std::endl;

    int num_detections;
    int num_values_per_detection;

    if (output_data.dims == 3) {
        int dim1 = output_data.size[1];
        int dim2 = output_data.size[2];

        std::cout << "[DEBUG] dim1=" << dim1 << ", dim2=" << dim2 << std::endl;

        if (dim1 < 100 && dim2 > 1000) {
            output_data = output_data.reshape(1, dim1);
            cv::transpose(output_data, output_data);
            num_detections = dim2;
            num_values_per_detection = dim1;
            std::cout << "[DEBUG] Transposed to [" << num_detections << ", "
                      << num_values_per_detection << "]" << std::endl;
        } else {
            output_data = output_data.reshape(1, dim1);
            num_detections = dim1;
            num_values_per_detection = dim2;
            std::cout << "[DEBUG] Reshaped to [" << num_detections << ", "
                      << num_values_per_detection << "]" << std::endl;
        }
    } else if (output_data.dims == 2) {
        num_detections = output_data.rows;
        num_values_per_detection = output_data.cols;
        std::cout << "[DEBUG] Already 2D: [" << num_detections << ", "
                  << num_values_per_detection << "]" << std::endl;
    } else {
        std::cerr << "ERROR: Unexpected output dimensions!" << std::endl;
        return detections;
    }

    bool is_single_class = (num_values_per_detection == 5);
    int num_classes = is_single_class ? 1 : (num_values_per_detection - 4);

    std::cout << "[DEBUG] Number of classes: " << num_classes << std::endl;
    std::cout << "[DEBUG] Format: " << (is_single_class ? "Single-class" : "Multi-class") << std::endl;
    std::cout << "[DEBUG] Processing " << num_detections << " detections..." << std::endl;

    int valid_detections = 0;

    for (int i = 0; i < num_detections; ++i) {
        float* data = output_data.ptr<float>(i);

        float x_center = data[0];
        float y_center = data[1];
        float width = data[2];
        float height = data[3];

        float max_score;
        int max_class_id = 0;

        if (is_single_class) {
            max_score = data[4];
        } else {
            float* class_scores = data + 4;
            max_score = -1.0f;
            max_class_id = -1;

            for (int c = 0; c < num_classes; c++) {
                if (class_scores[c] > max_score) {
                    max_score = class_scores[c];
                    max_class_id = c;
                }
            }
        }

        if (max_score < conf_threshold) {
            continue;
        }

        valid_detections++;

        if (valid_detections <= 3) {
            std::cout << "[DEBUG] Detection " << valid_detections << ":" << std::endl;
            std::cout << "  Raw coords: x=" << x_center << ", y=" << y_center
                      << ", w=" << width << ", h=" << height << std::endl;
            std::cout << "  Max score: " << max_score << ", class: " << max_class_id << std::endl;
        }

        // Check if normalized (0-1) or absolute (640x640 input space)
        bool is_normalized = (x_center <= 1.0f && y_center <= 1.0f &&
                             width <= 1.0f && height <= 1.0f);

        if (is_normalized) {
            // Normalized coordinates - scale directly to image size
            x_center *= img_width;
            y_center *= img_height;
            width *= img_width;
            height *= img_height;
        } else {
            // Absolute coordinates in input space (640x640)
            // Scale to original image dimensions
            x_center *= scale_x;
            y_center *= scale_y;
            width *= scale_x;
            height *= scale_y;
        }

        // Convert from center coords to corner coords
        int x = static_cast<int>(x_center - width / 2.0f);
        int y = static_cast<int>(y_center - height / 2.0f);
        int w = static_cast<int>(width);
        int h = static_cast<int>(height);

        // Clamp to image bounds
        x = std::max(0, std::min(x, img_width - 1));
        y = std::max(0, std::min(y, img_height - 1));
        w = std::max(1, std::min(w, img_width - x));
        h = std::max(1, std::min(h, img_height - y));

        if (valid_detections <= 3) {
            std::cout << "  Scaled box: x=" << x << ", y=" << y
                      << ", w=" << w << ", h=" << h << std::endl;
        }

        class_ids.push_back(max_class_id);
        confidences.push_back(max_score);
        boxes.push_back(cv::Rect(x, y, w, h));
    }

    std::cout << "[DEBUG] Found " << valid_detections << " detections above threshold" << std::endl;

    std::vector<int> nms_result;
    cv::dnn::NMSBoxes(boxes, confidences, conf_threshold, nms_threshold, nms_result);

    std::cout << "[DEBUG] After NMS: " << nms_result.size() << " detections" << std::endl;

    for (int idx : nms_result) {
        Detection result;
        result.class_id = class_ids[idx];
        result.confidence = confidences[idx];
        result.bounding_box = boxes[idx];
        result.class_name = "bumper";

        detections.push_back(result);
    }

    return detections;
}

int main() {
    try {
        std::cout << std::string(60, '=') << std::endl;
        std::cout << "YOLO BUMPER DETECTION" << std::endl;
        std::cout << std::string(60, '=') << std::endl;

        std::string model_path = "C:/Users/marcu/CLionProjects/robotvisiontest/modeltest/bumper_yolov10.onnx";

        std::cout << "\nLoading YOLO model..." << std::endl;
        std::cout << "  Model: " << model_path << std::endl;

        cv::dnn::Net net = cv::dnn::readNetFromONNX(model_path);

        if (net.empty()) {
            std::cerr << "ERROR: Failed to load model!" << std::endl;
            return -1;
        }

        net.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
        net.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);

        std::cout << "  ✓ Model loaded successfully!" << std::endl;

        std::string video_path = "C:/Users/marcu/CLionProjects/robotvisiontest/robotsecond.MP4";

        std::cout << "\nOpening video..." << std::endl;
        std::cout << "  Video: " << video_path << std::endl;

        cv::VideoCapture cap(video_path);
        if (!cap.isOpened()) {
            std::cerr << "ERROR: Failed to open video!" << std::endl;
            return -1;
        }

        int total_frames = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_COUNT));
        double fps = cap.get(cv::CAP_PROP_FPS);

        std::cout << "  ✓ Video opened" << std::endl;
        std::cout << "  Total frames: " << total_frames << std::endl;
        std::cout << "  FPS: " << fps << std::endl;

        const int INPUT_WIDTH = 640;
        const int INPUT_HEIGHT = 640;
        const float CONF_THRESHOLD = 0.1;   // Lower threshold to see more detections
        const float NMS_THRESHOLD = 0.45;

        std::cout << "\n" << std::string(60, '=') << std::endl;
        std::cout << "PROCESSING VIDEO" << std::endl;
        std::cout << std::string(60, '=') << std::endl;
        std::cout << "Confidence threshold: " << CONF_THRESHOLD << std::endl;
        std::cout << "Press ESC to exit\n" << std::endl;

        cv::Mat frame;
        int frame_count = 0;
        int detection_count = 0;
        bool first_frame = true;

        while (true) {
            if (!cap.read(frame)) {
                std::cout << "\nEnd of video reached." << std::endl;
                break;
            }

            if (frame.empty()) continue;

            frame_count++;

            // Process every 4th frame
            if (frame_count % 4 != 0) continue;

            std::cout << "\n--- Frame " << frame_count << "/" << total_frames << " ---" << std::endl;

            // Prepare input blob
            cv::Mat blob;
            cv::dnn::blobFromImage(
                frame,
                blob,
                1.0 / 255.0,
                cv::Size(INPUT_WIDTH, INPUT_HEIGHT),
                cv::Scalar(0, 0, 0),
                true,
                false
            );

            net.setInput(blob);

            // Run inference
            std::vector<cv::Mat> outputs;
            net.forward(outputs, net.getUnconnectedOutLayersNames());

            // Process outputs
            std::vector<Detection> detections = ProcessYoloOutput(
                outputs,
                frame.cols,
                frame.rows,
                INPUT_WIDTH,
                INPUT_HEIGHT,
                CONF_THRESHOLD,
                NMS_THRESHOLD
            );

            // Only show debug info for first frame
            first_frame = false;

            detection_count += detections.size();

            // Draw detections
            for (const auto& det : detections) {
                cv::rectangle(
                    frame,
                    det.bounding_box,
                    cv::Scalar(0, 255, 0),
                    3
                );

                std::stringstream ss;
                ss << det.class_name << ": "
                   << static_cast<int>(det.confidence * 100) << "%";
                std::string label = ss.str();

                int baseline;
                cv::Size label_size = cv::getTextSize(
                    label,
                    cv::FONT_HERSHEY_SIMPLEX,
                    0.6,
                    2,
                    &baseline
                );

                cv::Point label_origin(det.bounding_box.x, det.bounding_box.y - 10);

                cv::rectangle(
                    frame,
                    cv::Point(label_origin.x,
                             label_origin.y - label_size.height - baseline),
                    cv::Point(label_origin.x + label_size.width,
                             label_origin.y + baseline),
                    cv::Scalar(0, 255, 0),
                    cv::FILLED
                );

                cv::putText(
                    frame,
                    label,
                    label_origin,
                    cv::FONT_HERSHEY_SIMPLEX,
                    0.6,
                    cv::Scalar(0, 0, 0),
                    2
                );
            }

            cv::imshow("YOLO Bumper Detection", frame);

            int key = cv::waitKey(1);
            if (key == 27) {
                std::cout << "\n\nStopped by user." << std::endl;
                break;
            }
        }

        std::cout << "\n" << std::string(60, '=') << std::endl;
        std::cout << "SUMMARY" << std::endl;
        std::cout << std::string(60, '=') << std::endl;
        std::cout << "Frames processed: " << frame_count / 4 << std::endl;
        std::cout << "Total detections: " << detection_count << std::endl;
        std::cout << std::string(60, '=') << std::endl;

    } catch (const cv::Exception& e) {
        std::cerr << "\nOpenCV Error: " << e.what() << std::endl;
        return -1;
    } catch (const std::exception& e) {
        std::cerr << "\nError: " << e.what() << std::endl;
        return -1;
    }

    return 0;
}