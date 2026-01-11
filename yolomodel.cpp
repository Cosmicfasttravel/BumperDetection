#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <vector>
#include <string>
#include <chrono>

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
    int input_width,
    int input_height,
    float conf_threshold,
    float nms_threshold
) {
    std::vector<Detection> detections;
    std::vector<int> class_ids;
    std::vector<float> confidences;
    std::vector<cv::Rect> boxes;

    float scale_x = static_cast<float>(img_width) / input_width;
    float scale_y = static_cast<float>(img_height) / input_height;

    cv::Mat output_data = outputs[0];

    int num_detections;
    int num_values_per_detection;

    if (output_data.dims == 3) {
        int dim1 = output_data.size[1];
        int dim2 = output_data.size[2];

        if (dim1 < 100 && dim2 > 1000) {
            output_data = output_data.reshape(1, dim1);
            cv::transpose(output_data, output_data);
            num_detections = dim2;
            num_values_per_detection = dim1;
        } else {
            output_data = output_data.reshape(1, dim1);
            num_detections = dim1;
            num_values_per_detection = dim2;
        }
    } else if (output_data.dims == 2) {
        num_detections = output_data.rows;
        num_values_per_detection = output_data.cols;
    } else {
        std::cerr << "ERROR: Unexpected output dimensions!" << std::endl;
        return detections;
    }

    bool is_single_class = (num_values_per_detection == 5);
    int num_classes = is_single_class ? 1 : (num_values_per_detection - 4);

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

        bool is_normalized = (x_center <= 1.0f && y_center <= 1.0f &&
                             width <= 1.0f && height <= 1.0f);

        if (is_normalized) {
            x_center *= img_width;
            y_center *= img_height;
            width *= img_width;
            height *= img_height;
        } else {
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

        class_ids.push_back(max_class_id);
        confidences.push_back(max_score);
        boxes.push_back(cv::Rect(x, y, w, h));
    }

    std::vector<int> nms_result;
    cv::dnn::NMSBoxes(boxes, confidences, conf_threshold, nms_threshold, nms_result);

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

// Function to test and select the best available backend
void setupGPUBackend(cv::dnn::Net& net) {
    std::cout << "\n[GPU DETECTION]" << std::endl;
    std::cout << "Testing available GPU backends..." << std::endl;

    // Test OpenCL
    try {
        cv::dnn::Net test_net = net;
        test_net.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
        test_net.setPreferableTarget(cv::dnn::DNN_TARGET_OPENCL);

        std::cout << "  OpenCL is available!" << std::endl;
        net.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
        net.setPreferableTarget(cv::dnn::DNN_TARGET_OPENCL);
        std::cout << "  Using OpenCL for GPU acceleration" << std::endl;
        return;
    } catch (...) {
        std::cout << "  OpenCL not available" << std::endl;
    }

    // Test Vulkan
    try {
        cv::dnn::Net test_net = net;
        test_net.setPreferableBackend(cv::dnn::DNN_BACKEND_VKCOM);
        test_net.setPreferableTarget(cv::dnn::DNN_TARGET_VULKAN);

        std::cout << "  Vulkan is available!" << std::endl;
        net.setPreferableBackend(cv::dnn::DNN_BACKEND_VKCOM);
        net.setPreferableTarget(cv::dnn::DNN_TARGET_VULKAN);
        std::cout << "  Using Vulkan for GPU acceleration" << std::endl;
        return;
    } catch (...) {
        std::cout << "  Vulkan not available" << std::endl;
    }

    // Test OpenCL with FP16
    try {
        cv::dnn::Net test_net = net;
        test_net.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
        test_net.setPreferableTarget(cv::dnn::DNN_TARGET_OPENCL_FP16);

        std::cout << "  OpenCL FP16 is available!" << std::endl;
        net.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
        net.setPreferableTarget(cv::dnn::DNN_TARGET_OPENCL_FP16);
        std::cout << "  Using OpenCL FP16 for GPU acceleration" << std::endl;
        return;
    } catch (...) {
        std::cout << "  OpenCL FP16 not available" << std::endl;
    }

    // Fallback to CPU
    std::cout << "  No GPU backends available, using CPU" << std::endl;
    net.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
    net.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);
}

int main() {
    try {
        std::cout << std::string(60, '=') << std::endl;
        std::cout << "YOLO BUMPER DETECTION (GPU ACCELERATED)" << std::endl;
        std::cout << std::string(60, '=') << std::endl;

        std::string model_path = "C:/Users/marcu/CLionProjects/robotvisiontest/modeltest/bumper_yolov10.onnx";

        std::cout << "\nLoading YOLO model..." << std::endl;
        std::cout << "  Model: " << model_path << std::endl;

        cv::dnn::Net net = cv::dnn::readNetFromONNX(model_path);

        if (net.empty()) {
            std::cerr << "ERROR: Failed to load model!" << std::endl;
            return -1;
        }

        // Automatically detect and set up best GPU backend
        setupGPUBackend(net);

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
        const float CONF_THRESHOLD = 0.1;
        const float NMS_THRESHOLD = 0.45;

        std::cout << "\n" << std::string(60, '=') << std::endl;
        std::cout << "PROCESSING VIDEO" << std::endl;
        std::cout << std::string(60, '=') << std::endl;
        std::cout << "Confidence threshold: " << CONF_THRESHOLD << std::endl;
        std::cout << "Press ESC to exit\n" << std::endl;

        cv::Mat frame;
        int frame_count = 0;
        int detection_count = 0;

        auto start_time = std::chrono::high_resolution_clock::now();

        while (true) {
            if (!cap.read(frame)) {
                std::cout << "\nEnd of video reached." << std::endl;
                break;
            }

            if (frame.empty()) continue;

            frame_count++;

            // Process every frame with GPU acceleration
            if (frame_count % 4 != 0) continue;

            if (frame_count % 100 == 0) {
                auto current_time = std::chrono::high_resolution_clock::now();
                auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
                    current_time - start_time).count() / 1000.0;
                double current_fps = frame_count / elapsed;
                std::cout << "Frame " << frame_count << "/" << total_frames
                          << " - FPS: " << std::fixed << std::setprecision(1)
                          << current_fps << std::endl;
            }

            cv::UMat frameUMat;
            frame.copyTo(frameUMat); // Upload frame to GPU memory immediately

            cv::Mat blob;
            cv::dnn::blobFromImage(
                frameUMat, // Use UMat here
                blob,
                1.0 / 255.0,
                cv::Size(INPUT_WIDTH, INPUT_HEIGHT),
                cv::Scalar(0, 0, 0),
                true,
                false
            );

            net.setInput(blob);
            net.setPreferableTarget(cv::dnn::DNN_TARGET_OPENCL);
            std::vector<cv::Mat> outputs;
            net.forward(outputs, net.getUnconnectedOutLayersNames());

            std::vector<Detection> detections = ProcessYoloOutput(
                outputs,
                frame.cols,
                frame.rows,
                INPUT_WIDTH,
                INPUT_HEIGHT,
                CONF_THRESHOLD,
                NMS_THRESHOLD
            );

            detection_count += detections.size();

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

            cv::imshow("YOLO Bumper Detection (GPU)", frame);

            int key = cv::waitKey(1);
            if (key == 27) {
                std::cout << "\n\nStopped by user." << std::endl;
                break;
            }
        }

        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(
            end_time - start_time);

        std::cout << "\n" << std::string(60, '=') << std::endl;
        std::cout << "SUMMARY" << std::endl;
        std::cout << std::string(60, '=') << std::endl;
        std::cout << "Frames processed: " << frame_count << std::endl;
        std::cout << "Total detections: " << detection_count << std::endl;
        std::cout << "Processing time: " << duration.count() / 1000.0 << " seconds" << std::endl;
        std::cout << "Average FPS: " << std::fixed << std::setprecision(2)
                  << frame_count / (duration.count() / 1000.0) << std::endl;
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