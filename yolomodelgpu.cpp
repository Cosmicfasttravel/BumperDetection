#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/core/ocl.hpp>
#include <opencv2/dnn.hpp>
#include <vector>
#include <string>
#include <chrono>
#include <iomanip>

#include "edgeDetection.h"
#include "GPU.h"

// Include CUDA module if available
#ifdef HAVE_OPENCV_CUDNN
#include <opencv2/core/cuda.hpp>
#endif


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
        boxes.emplace_back(x, y, w, h);
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



int main() {
    try {
        std::cout << std::string(60, '=') << std::endl;
        std::cout << "YOLO BUMPER DETECTION" << std::endl;
        std::cout << std::string(60, '=') << std::endl;

        // Check OpenCV build info
        std::cout << "\n[OpenCV Build Info]" << std::endl;
        std::cout << "OpenCV version: " << CV_VERSION << std::endl;

        #ifdef HAVE_OPENCV_CUDNN
        std::cout << "CUDA DNN support: YES" << std::endl;
        #else
        std::cout << "CUDA DNN support: NO" << std::endl;
        #endif

        std::cout << "OpenCL support: " << (cv::ocl::haveOpenCL() ? "YES" : "NO") << std::endl;

        std::string model_path = "C:/Users/marcu/CLionProjects/robotvisiontest/modeltest/bumper_yolov10.onnx"; //v10 = compact, v9 = tiny

        std::cout << "\n[Loading Model]" << std::endl;
        std::cout << "  Model: " << model_path << std::endl;

        cv::dnn::Net net = cv::dnn::readNetFromONNX(model_path);

        if (net.empty()) {
            std::cerr << "ERROR: Failed to load model!" << std::endl;
            return -1;
        }

        // Setup best available backend
        std::string backend = gpu::setupGPUBackend(net);

        std::cout << "\nModel loaded with " << backend << " backend" << std::endl;

        std::string video_path = "C:/Users/marcu/CLionProjects/robotvisiontest/robotsecond.MP4";

        std::cout << "\n[Opening Video]" << std::endl;

        cv::VideoCapture cap(video_path);
        if (!cap.isOpened()) {
            std::cerr << "ERROR: Failed to open video!" << std::endl;
            return -1;
        }

        int total_frames = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_COUNT));
        double fps = cap.get(cv::CAP_PROP_FPS);
        int frame_width = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_WIDTH));
        int frame_height = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_HEIGHT));

        std::cout << "  Resolution: " << frame_width << "x" << frame_height << std::endl;
        std::cout << "  Total frames: " << total_frames << std::endl;
        std::cout << "  FPS: " << fps << std::endl;

        constexpr int INPUT_WIDTH = 640;
        constexpr int INPUT_HEIGHT = 640;
        constexpr float CONF_THRESHOLD = 0.5;

        // Adjust frame skip based on backend
        int frame_skip = 1;  // Process every frame with CUDA
        if (backend == "OpenCL" || backend == "OpenCL_FP16") {
            frame_skip = 5;  // Every 5th frame for OpenCL
        } else if (backend == "CPU") {
            frame_skip = 10; // Every 10th frame for CPU
        }

        std::cout << "\n" << std::string(60, '=') << std::endl;
        std::cout << "PROCESSING VIDEO" << std::endl;
        std::cout << std::string(60, '=') << std::endl;
        std::cout << "Backend: " << backend << std::endl;
        std::cout << "Input size: " << INPUT_WIDTH << "x" << INPUT_HEIGHT << std::endl;
        std::cout << "Frame skip: every " << frame_skip << " frame(s)" << std::endl;
        std::cout << "Confidence threshold: " << CONF_THRESHOLD << std::endl;
        std::cout << "Press ESC to exit\n" << std::endl;

        cv::Mat frame, blankFrame;
        int frame_count = 0;
        int processed_count = 0;
        int detection_count = 0;

        auto start_time = std::chrono::high_resolution_clock::now();

        // Timing variables
        long long total_blob_time = 0;
        long long total_inference_time = 0;
        long long total_postprocess_time = 0;

        while (true) {

            constexpr float NMS_THRESHOLD = 0.45;
            if (!cap.read(frame)) {
                std::cout << "\nEnd of video reached." << std::endl;
                break;
            }
            if (frame.empty()) continue;

            frame_count++;

            // Process based on frame_skip
            if (frame_count % frame_skip != 0) {
                // Still display the frame
                cv::imshow("YOLO Bumper Detection", frame);
                if (cv::waitKey(1) == 27) break;
                continue;
            }
            processed_count++;

            //Create a blank frame that doesn't contain the rectangles
            blankFrame = frame.clone();

            // BLOB CREATION with timing
            auto blob_start = std::chrono::high_resolution_clock::now();

            cv::Mat blob;
            if (backend == "OpenCL" || backend == "OpenCL_FP16") {
                cv::UMat frameUMat;
                frame.copyTo(frameUMat);
                cv::dnn::blobFromImage(frameUMat, blob, 1.0 / 255.0,
                    cv::Size(INPUT_WIDTH, INPUT_HEIGHT), cv::Scalar(0, 0, 0), true, false);
            } else {
                cv::dnn::blobFromImage(frame, blob, 1.0 / 255.0,
                    cv::Size(INPUT_WIDTH, INPUT_HEIGHT), cv::Scalar(0, 0, 0), true, false);
            }

            auto blob_end = std::chrono::high_resolution_clock::now();
            total_blob_time += std::chrono::duration_cast<std::chrono::milliseconds>(blob_end - blob_start).count();

            // INFERENCE with timing
            auto inference_start = std::chrono::high_resolution_clock::now();

            net.setInput(blob);
            std::vector<cv::Mat> outputs;
            net.forward(outputs, net.getUnconnectedOutLayersNames());

            auto inference_end = std::chrono::high_resolution_clock::now();
            total_inference_time += std::chrono::duration_cast<std::chrono::milliseconds>(inference_end - inference_start).count();

            // POST-PROCESSING with timing
            auto postprocess_start = std::chrono::high_resolution_clock::now();

            std::vector<Detection> detections = ProcessYoloOutput(
                outputs, frame.cols, frame.rows,
                INPUT_WIDTH, INPUT_HEIGHT,
                CONF_THRESHOLD, NMS_THRESHOLD
            );

            auto postprocess_end = std::chrono::high_resolution_clock::now();
            total_postprocess_time += std::chrono::duration_cast<std::chrono::milliseconds>(postprocess_end - postprocess_start).count();

            detection_count += detections.size();


            // Draw detections
            for (const auto& det : detections) {
                cv::rectangle(frame, det.bounding_box, cv::Scalar(0, 255, 0), 3);

                std::stringstream ss;
                ss << det.class_name << ": " << static_cast<int>(det.confidence * 100) << "%";
                std::string label = ss.str();

                int baseline;
                cv::Size label_size = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.6, 2, &baseline);
                cv::Point label_origin(det.bounding_box.x, det.bounding_box.y - 10);

                cv::rectangle(frame,
                    cv::Point(label_origin.x, label_origin.y - label_size.height - baseline),
                    cv::Point(label_origin.x + label_size.width, label_origin.y + baseline),
                    cv::Scalar(0, 255, 0), cv::FILLED);

                cv::putText(frame, label, label_origin, cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 0, 0), 2);
            }

            // Display backend on frame
            cv::putText(frame, "Backend: " + backend, cv::Point(10, 30),
                cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 255, 255), 2);

            // Show progress every 25 processed frames
            if (processed_count % 25 == 0) {
                auto current_time = std::chrono::high_resolution_clock::now();
                auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
                    current_time - start_time).count() / 1000.0;
                double current_fps = processed_count / elapsed;

                std::cout << "Frame " << frame_count << "/" << total_frames
                          << " (Processed: " << processed_count << ")"
                          << " - FPS: " << std::fixed << std::setprecision(1) << current_fps << std::endl;
            }
            det::detectEdgesBumper(blankFrame, frame, detections);

            int key = cv::waitKey(1);
            bool paused = false;
            if (key == 27) {
                std::cout << "\nStopped by user." << std::endl;
                break;
            }
            if (key == 112) paused = true;
            if (paused) cv::waitKey(-1);
        }

        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
        std::cout << "\n" << std::string(60, '=') << std::endl;
        std::cout << "PERFORMANCE SUMMARY" << std::endl;
        std::cout << std::string(60, '=') << std::endl;
        std::cout << "Backend: " << backend << std::endl;
        std::cout << "Total frames: " << frame_count << std::endl;
        std::cout << "Frames processed: " << processed_count << std::endl;
        std::cout << "Total detections: " << detection_count << std::endl;
        std::cout << "Processing time: " << std::fixed << std::setprecision(2)
                  << duration.count() / 1000.0 << " seconds" << std::endl;
        std::cout << "Average FPS: " << std::fixed << std::setprecision(2)
                  << processed_count / (duration.count() / 1000.0) << std::endl;
        std::cout << "\nTiming Breakdown (per frame):" << std::endl;
        std::cout << "  Blob creation: " << (total_blob_time / processed_count) << "ms" << std::endl;
        std::cout << "  Inference: " << (total_inference_time / processed_count) << "ms" << std::endl;
        std::cout << "  Post-processing: " << (total_postprocess_time / processed_count) << "ms" << std::endl;
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