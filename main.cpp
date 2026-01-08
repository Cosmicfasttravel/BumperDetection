#include <iostream>
#include <opencv2/opencv.hpp>
#define ORT_API_MANUAL_INIT
#include <onnxruntime_cxx_api.h>
#include <vector>
#include <string>
#include <array>
#include <map>

cv::Mat letterboxToSquare(const cv::Mat &src, int size = 224) {
    int w = src.cols;
    int h = src.rows;

    float scale = std::min(
        size / (float) w,
        size / (float) h
    );

    int newW = int(w * scale);
    int newH = int(h * scale);

    cv::Mat resized;
    cv::resize(src, resized, cv::Size(newW, newH));

    cv::Mat output = cv::Mat::zeros(size, size, src.type());
    int x = (size - newW) / 2;
    int y = (size - newH) / 2;

    resized.copyTo(output(cv::Rect(x, y, newW, newH)));
    return output;
}

int main() {
    try {
        // Initialize ONNX Runtime with API version 17
        Ort::InitApi(OrtGetApiBase()->GetApi(17));

        Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "BumperDetect");
        Ort::SessionOptions session_options;
        session_options.SetIntraOpNumThreads(1);

        std::wstring model_path = L"C:/Users/marcu/CLionProjects/robotvisiontest/modeltest/bumper_classifier.ONNX";
        Ort::Session session(env, model_path.c_str(), session_options);

        // Print input/output info for debugging
        Ort::AllocatorWithDefaultOptions allocator;
        size_t num_input_nodes = session.GetInputCount();
        size_t num_output_nodes = session.GetOutputCount();

        std::cout << "Model has " << num_input_nodes << " inputs and "
                << num_output_nodes << " outputs\n";

        // Get actual input/output names
        auto input_name_allocated = session.GetInputNameAllocated(0, allocator);
        auto output_name_allocated = session.GetOutputNameAllocated(0, allocator);
        const char *input_name = input_name_allocated.get();
        const char *output_name = output_name_allocated.get();

        std::cout << "Input name: " << input_name << "\n";
        std::cout << "Output name: " << output_name << "\n";

        // Get input shape
        auto input_shape_info = session.GetInputTypeInfo(0).GetTensorTypeAndShapeInfo();
        auto input_dims = input_shape_info.GetShape();
        std::cout << "Input shape: ";
        for (auto dim: input_dims) std::cout << dim << " ";
        std::cout << "\n";

        Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

        cv::Ptr<cv::BackgroundSubtractor> bg = cv::createBackgroundSubtractorKNN(500, 400, false);
        cv::VideoCapture cap("C:/Users/marcu/CLionProjects/robotvisiontest/robotsecond.MP4");
        if (!cap.isOpened()) {
            std::cerr << "Failed to open video\n";
            return -1;
        }

        cv::Mat frame;
        int frame_count = 0;
        int actual_frame_count = 0;

        std::map<int, cv::Rect> previous_detections;
        const int DETECTION_MEMORY = 20 * 4;

        while (true) {
            frame_count++;
            if (!cap.read(frame)) break;
            actual_frame_count++;
            if (!cap.read(frame)) break;
            actual_frame_count++;
            if (!cap.read(frame)) break;
            actual_frame_count++;
            if (!cap.read(frame)) break;
            actual_frame_count++;
            if (frame.empty()) continue;

            cv::Mat fgMask;

            bg->apply(frame, fgMask);
            cv::threshold(fgMask, fgMask, 200, 255, cv::THRESH_BINARY);
            cv::Mat fgKernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(5, 5));
            cv::morphologyEx(fgMask, fgMask, cv::MORPH_OPEN, fgKernel);
            cv::morphologyEx(fgMask, fgMask, cv::MORPH_CLOSE, fgKernel);



            std::cout << "Processing frame " << frame_count << "\n";

            cv::Mat hsv; //hsv
            // cv::GaussianBlur(frame, frame, cv::Size(0, 0), 0);
            cv::cvtColor(frame, hsv, cv::COLOR_BGR2HSV);

            cv::imshow("hsv", hsv);

            cv::Mat bMask, oMask, rMask;
            cv::inRange(hsv,
                        cv::Scalar(100, 120, 70), // Min HSV for blue
                        cv::Scalar(130, 255, 255), // Max HSV for blue
                        bMask);
            cv::inRange(hsv,
                        cv::Scalar(0, 120, 70), // Min HSV for red
                        cv::Scalar(10, 255, 255), // Max HSV for red
                        rMask);

            cv::bitwise_or(bMask, rMask, bMask);

            cv::bitwise_and(bMask, fgMask, oMask);

            cv::imshow("hsvmasked", bMask);
            cv::imshow("final", oMask);

            std::vector<std::vector<cv::Point> > contours;
            cv::findContours(oMask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

            cv::morphologyEx(oMask, oMask, cv::MORPH_OPEN, fgKernel);
            cv::morphologyEx(oMask, oMask, cv::MORPH_CLOSE, fgKernel);

            cv::Mat contourView = frame.clone();
            std::vector<std::vector<cv::Point> > validContours;

            cv::Mat contourViewVis = frame.clone();

            for (size_t i = 0; i < contours.size(); ++i) {
                auto &c = contours[i];
                double area = cv::contourArea(c);
                if (area < 1000) continue;
                if (area > 30000) continue;

                std::cout << "  Contour " << i << " area: " << area << "\n";

                cv::Rect bbox = cv::boundingRect(c);
                bbox = bbox & cv::Rect(0, 0, frame.cols, frame.rows);

                if (bbox.width <= 0 || bbox.height <= 0) continue;
                float aspect_ratio = (float) bbox.width / bbox.height;
                if (aspect_ratio < 0.5 || aspect_ratio > 15.0) {
                    std::cout << "  Skipped: bad aspect ratio " << aspect_ratio << "\n";
                    continue;
                }
                std::vector<cv::Point> hull;
                cv::convexHull(c, hull);
                double hull_area = cv::contourArea(hull);
                double solidity = area / hull_area;

                if (solidity < 0.25 || solidity > 0.85) {
                    std::cout << "  Not solid enough: " << solidity << "\n";
                    continue;
                }
                std::cout << "  Solidity: " << solidity << "\n";
                if (bbox.width < 60) {
                    std::cout << "  Too narrow: " << bbox.width << "px\n";
                    continue;
                }
                bool near_previous = false;
                for (auto& prev : previous_detections) {
                    cv::Rect& prev_bbox = prev.second;

                    // Calculate overlap
                    cv::Rect intersection = bbox & prev_bbox;
                    float overlap = (float)intersection.area() / (float)bbox.area();

                    if (overlap < 0.25) {
                        std::cout << "Overlap: " << overlap << '\n';
                        near_previous = true;
                        cv::rectangle(contourView, bbox, cv::Scalar(255, 255, 255), cv::FILLED);
                    }
                    else {
                        break;
                    }
                }

                cv::Mat roiFgMask = oMask(bbox).clone();

                cv::Mat roiMasked;
                cv::threshold(roiFgMask, roiFgMask, 128, 255, cv::THRESH_BINARY);

                cv::Mat roiInput = letterboxToSquare(roiFgMask, 224);

                cv::Mat roiFloat;
                roiInput.convertTo(roiFloat, CV_32FC1, 1.0 / 255.0);

                if (!roiFloat.isContinuous()) {
                    roiFloat = roiFloat.clone();
                }

                std::vector<float> input_tensor_values(1 * 1 * 224 * 224);

                int idx = 0;
                for (int h = 0; h < 224; ++h) {
                    for (int w = 0; w < 224; ++w) {
                        input_tensor_values[idx++] = roiFloat.at<float>(h, w);
                    }
                }

                std::array<int64_t, 4> input_shape{1, 1, 224, 224};

                Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
                    memory_info,
                    input_tensor_values.data(),
                    input_tensor_values.size(),
                    input_shape.data(),
                    input_shape.size()
                );

                const char *input_names[] = {input_name};
                const char *output_names[] = {output_name};

                auto output_tensors = session.Run(
                    Ort::RunOptions{nullptr},
                    input_names,
                    &input_tensor,
                    1,
                    output_names,
                    1
                );

                float *output = output_tensors.front().GetTensorMutableData<float>();

                auto output_shape = output_tensors.front().GetTensorTypeAndShapeInfo().GetShape();

                size_t output_size = 1;
                for (auto dim: output_shape) output_size *= dim;

                if (output_size >= 2) {
                    float logit0 = output[0];
                    float logit1 = output[1];

                    float max_logit = std::max(logit0, logit1);
                    float exp0 = std::exp(logit0 - max_logit);
                    float exp1 = std::exp(logit1 - max_logit);
                    float sum_exp = exp0 + exp1;

                    float prob0 = exp0 / sum_exp;
                    float bumper_prob = exp1 / sum_exp;

                    std::cout << "    Logits: [" << logit0 << ", " << logit1 << "]\n";
                    std::cout << "    Probabilities: [" << prob0 << ", " << bumper_prob << "]\n";
                    std::cout << "    Bumper probability: " << bumper_prob << "\n";




                    if (prob0 >= 0 && prob0 <= 0.25) {
                        cv::rectangle(contourView, bbox, cv::Scalar(255, 0, 0), 1);
                        validContours.push_back(c);
                        std::string filename = "C:/Users/marcu/CLionProjects/robotvisiontest/false_positives/unlikely_"
                                               +
                                               std::to_string(frame_count) + "_" +
                                               std::to_string(i) + ".jpg";

                        bool success = cv::imwrite(filename, roiInput);
                        if (success) {
                            std::cout << "    Saved: " << filename << "\n";
                        } else {
                            std::cerr << "    Failed to save: " << filename << "\n";
                        }
                    } else if ((prob0 > 0.25 && prob0 <= 0.75)) {
                        cv::rectangle(contourView, bbox, cv::Scalar(0, 0, 255), 3);
                        validContours.push_back(c);
                        std::string filename = "C:/Users/marcu/CLionProjects/robotvisiontest/false_positives/likely_" +
                                               std::to_string(frame_count) + "_" +
                                               std::to_string(i) + ".jpg";

                        bool success = cv::imwrite(filename, roiInput);
                        if (success) {
                            std::cout << "    Saved: " << filename << "\n";
                        } else {
                            std::cerr << "    Failed to save: " << filename << "\n";
                        }
                    } else if (prob0 > 0.75){
                        cv::rectangle(contourView, bbox, cv::Scalar(0, 255, 0), 5);
                        validContours.push_back(c);
                        std::string filename =
                                "C:/Users/marcu/CLionProjects/robotvisiontest/false_positives/almostabsolute_" +
                                std::to_string(frame_count) + "_" +
                                std::to_string(i) + ".jpg";

                        bool success = cv::imwrite(filename, roiInput);
                        if (success) {
                            std::cout << "    Saved: " << filename << "\n";
                        } else {
                            std::cerr << "    Failed to save: " << filename << "\n";
                        }
                    }

                } else {
                }
                previous_detections[actual_frame_count] = bbox;
            }

            // Clean up old detections
            for (auto it = previous_detections.begin(); it != previous_detections.end();) {
                if (actual_frame_count - it->first > DETECTION_MEMORY) {
                    it = previous_detections.erase(it);
                } else {
                    ++it;
                }
            }

            cv::drawContours(contourView, validContours, -1, cv::Scalar(0, 255, 255), 2);
            cv::imshow("Bumpers", contourView);
            if (cv::waitKey(15) == 27) break; // ESC to exit
            // cv::waitKey(0);
        }
    } catch (const Ort::Exception &e) {
        std::cerr << "ONNX Runtime error: " << e.what() << "\n";
        return -1;
    } catch (const cv::Exception &e) {
        std::cerr << "OpenCV error: " << e.what() << "\n";
        return -1;
    } catch (const std::exception &e) {
        std::cerr << "Error: " << e.what() << "\n";
        return -1;
    }


    return 0;
}
