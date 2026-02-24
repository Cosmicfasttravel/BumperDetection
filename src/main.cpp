#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/core/ocl.hpp>
#include <opencv2/dnn.hpp>
#include <vector>
#include <string>
#include <chrono>
#include <iomanip>
#include "robotTracking.h"
#include "setupGPU.h"

using Clock = std::chrono::high_resolution_clock;

// Include CUDA module if available
#ifdef HAVE_OPENCV_CUDNN
#endif


std::vector<Detection> ProcessYoloOutput(
    const std::vector<cv::Mat> &outputs,
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

    float scale_x = static_cast<float>(img_width) / static_cast<float>(input_width);
    float scale_y = static_cast<float>(img_height) / static_cast<float>(input_height);

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
        auto *data = output_data.ptr<float>(i);

        float x_center = data[0];
        float y_center = data[1];
        float width = data[2];
        float height = data[3];

        float max_score;
        int max_class_id = 0;

        if (is_single_class) {
            max_score = data[4];
        } else {
            float *class_scores = data + 4;
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
            x_center *= static_cast<float>(img_width);
            y_center *= static_cast<float>(img_height);
            width *= static_cast<float>(img_width);
            height *= static_cast<float>(img_height);
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

        class_ids.emplace_back(max_class_id);
        confidences.emplace_back(max_score);
        boxes.emplace_back(x, y, w, h);
    }

    std::vector<int> nms_result;
    cv::dnn::NMSBoxes(boxes, confidences, conf_threshold, nms_threshold, nms_result);

    for (int idx: nms_result) {
        Detection result;
        result.class_id = class_ids[idx];
        result.confidence = confidences[idx];
        result.bounding_box = boxes[idx];
        result.class_name = "bumper";

        detections.emplace_back(result);
    }

    return detections;
}


int main() {
    std::string teamNumbers[5] = {"1306", "2560", "2457", "4959", "118"};
    //for robot cropped "3928", "2560", "2457", "4959", "118"

    // std::cout << "Input team numbers (5): " << std::endl;
    // std::cout << "1: ";
    // std::cin >> teamNumbers[0];
    // std::cout << std::endl << "2: ";
    // std::cin >> teamNumbers[1];
    // std::cout << std::endl << "3: ";
    // std::cin >> teamNumbers[2];
    // std::cout << std::endl << "4: ";
    // std::cin >> teamNumbers[3];
    // std::cout << std::endl << "5: ";
    // std::cin >> teamNumbers[4];

    for (auto &teamNumber: teamNumbers) {
        if (teamNumbers->length() >= 5 || teamNumbers->length() <= 1) {
            teamNumber = "";
        }
    }

    try {
        std::string model_path = "C:/Users/marcu/CLionProjects/robotvisiontest/modeltest/bumper_yolov9.onnx";

        cv::dnn::Net net = cv::dnn::readNetFromONNX(model_path);

        if (net.empty()) {
            return -1;
        }
        // Setup best available backend
        std::string backend = setupGPUBackend(net);

        std::string video_path = "C:/Users/marcu/CLionProjects/robotvisiontest/vids/5ft.MP4";

        cv::VideoCapture cap(video_path);
        if (!cap.isOpened()) {
            return -1;
        }

        // Adjust frame skip based on backend
        int frame_skip = 1;

        cv::Mat frame, blankFrame;
        int frame_count = 0;
        int processed_count = 0;
        int detection_count = 0;

        auto start_time = std::chrono::high_resolution_clock::now();

        // Timing variables
        long long total_blob_time = 0;
        long long total_inference_time = 0;
        long long total_postprocess_time = 0;

        bool paused = false;
        int waitTime = 1;

        static auto prev_frame_time = Clock::now();

        startOCR();
        while (true) {
            auto frame_start = Clock::now();

            constexpr int INPUT_HEIGHT = 640;
            constexpr int INPUT_WIDTH = 640;
            constexpr float CONF_THRESHOLD = 0.75;
            constexpr float NMS_THRESHOLD = 0.45;
            if (!cap.read(frame)) {
                break;
            }
            if (frame.empty()) continue;

            frame_count++;

            // Process based on frame_skip
            if (frame_count % frame_skip != 0) {
                if (cv::waitKey(1) == 27) break;
                continue;
            }
            processed_count++;

            //Create a blank frame that doesn't contain the rectangles
            blankFrame = frame.clone();

            auto blob_start = std::chrono::high_resolution_clock::now();

            cv::Mat blob;

            cv::dnn::blobFromImage(frame, blob, 1.0 / 255.0,
                                   cv::Size(INPUT_WIDTH, INPUT_HEIGHT), cv::Scalar(0, 0, 0), true, false);

            auto blob_end = std::chrono::high_resolution_clock::now();
            total_blob_time += std::chrono::duration_cast<std::chrono::milliseconds>(blob_end - blob_start).count();

            auto inference_start = std::chrono::high_resolution_clock::now();

            net.setInput(blob);
            std::vector<cv::Mat> outputs;
            net.forward(outputs, net.getUnconnectedOutLayersNames());

            auto inference_end = std::chrono::high_resolution_clock::now();
            total_inference_time += std::chrono::duration_cast<std::chrono::milliseconds>(
                inference_end - inference_start).count();


            auto postprocess_start = std::chrono::high_resolution_clock::now();

            std::vector<Detection> detections = ProcessYoloOutput(
                outputs, frame.cols, frame.rows,
                INPUT_WIDTH, INPUT_HEIGHT,
                CONF_THRESHOLD, NMS_THRESHOLD
            );

            detection_count += static_cast<int>(detections.size());

            auto frame_end = Clock::now();
            using FrameDuration = std::chrono::duration<double>;
            auto delta = FrameDuration(frame_end - prev_frame_time).count(); // seconds
            std::stringstream ss;
            ss << std::fixed << std::setprecision(2) << "FPS: " << (1.0 / delta);

            cv::putText(frame, ss.str(), cv::Point(10, 50),
                        cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(255, 0, 255), 2);
            for (auto& det : detections) {
                cv::rectangle(frame, det.bounding_box, cv::FILLED);
            }
            analyzeDetections(blankFrame, teamNumbers, frame, detections);

            int key = cv::waitKey(waitTime);
            if (key == 27) {
                break;
            }
            if (key == 112) paused = !paused;
            if (paused) waitTime = -1;
            else waitTime = 1;

            prev_frame_time = frame_start;

            auto postprocess_end = std::chrono::high_resolution_clock::now();
            total_postprocess_time += std::chrono::duration_cast<std::chrono::milliseconds>(
                postprocess_end - postprocess_start).count();
        }

        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
        std::cout << "Processing time: " << std::fixed << std::setprecision(2)
                << static_cast<double>(duration.count()) / 1000.0 << " seconds" << std::endl;
        std::cout << "Average FPS: " << std::fixed << std::setprecision(2)
                << processed_count / (static_cast<double>(duration.count()) / 1000.0) << std::endl;
        std::cout << "\nTiming Breakdown (per frame):" << std::endl;
        std::cout << "  Blob creation: " << (total_blob_time / processed_count) << "ms" << std::endl;
        std::cout << "  Inference: " << (total_inference_time / processed_count) << "ms" << std::endl;
        std::cout << "  Post-processing: " << (total_postprocess_time / processed_count) << "ms" << std::endl;
    } catch (const cv::Exception &e) {
        return -1;
    } catch (const std::exception &e) {
        return -1;
    }

    endOCR();

    return 0;
}
