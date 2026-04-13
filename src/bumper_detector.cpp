#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/core/ocl.hpp>
#include <vector>
#include <ostream>
#include <string>
#include <chrono>
#include <iomanip>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#ifndef WIN32
#include "rknn_api.h"
#endif
#include "debug_log.h"
#include "analyze_detections.h"
#include <thread>
#include <atomic>
#include <mutex>
#include "tesseract/baseapi.h"
#include "config_extraction.h"

cv::Mat latestFrame;
std::mutex frameMutex;
std::atomic<bool> capturing(true);

void captureThread(cv::VideoCapture &cap)
{
    cv::Mat frame;
    while (capturing)
    {
        cap.read(frame);
        if (!frame.empty())
        {
            std::lock_guard<std::mutex> lock(frameMutex);
            latestFrame = frame.clone();
        }
    }
}

using Clock = std::chrono::high_resolution_clock;

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
        result.bounding_box = boxes[idx];
        if (result.bounding_box.width >= 450 || result.bounding_box.width <= 50)
        {
            continue;
        }

        detections.emplace_back(result);
    }

    return detections;
}

int run()
{
    extract();
    const Config &config = getConfig();

    initLogger();

    try
    {
#ifndef WIN32
        std::string model_path = config.input_paths.rknn_path;
        FILE *fp = fopen(model_path.c_str(), "rb");
        if (!fp)
        {
            logger->error("Failed to open model");
            return -1;
        }
        fseek(fp, 0, SEEK_END);
        long model_size = ftell(fp);
        rewind(fp);
        void *model_data = malloc(model_size);
        fread(model_data, 1, model_size, fp);
        fclose(fp);

        rknn_context ctx;
        int ret = rknn_init(&ctx, model_data, model_size, 0, nullptr);

        free(model_data);
        if (ret != 0)
        {
            logger->warn("Failed to init RKNN");
            return -1;
        }
        rknn_core_mask core_mask = RKNN_NPU_CORE_0_1_2;
        rknn_set_core_mask(ctx, core_mask);
        logger->info("RKNN Loaded sucessfully");
#else
        std::string model_path = config.input_paths.onnx_path;
        cv::dnn::Net net = cv::dnn::readNetFromONNX(model_path);
        if (net.empty())
        {
            return -1;
        }
#endif

        cv::VideoCapture cap;
        std::string video_path = config.input_paths.video_path;
#ifdef WIN32
        if (config.modes.video)
            cap.open(video_path);
        else
            cap.open(0, cv::CAP_DSHOW);
#else
        if (config.modes.video)
            cap.open(video_path);
        else
            cap.open(0, cv::CAP_V4L2);
#endif

        cap.set(cv::CAP_PROP_FOURCC, cv::VideoWriter::fourcc('M', 'J', 'P', 'G'));
        cap.set(cv::CAP_PROP_FRAME_WIDTH, config.screen.width);
        cap.set(cv::CAP_PROP_FRAME_HEIGHT, config.screen.height);
        cap.set(cv::CAP_PROP_FPS, 60);
        cap.set(cv::CAP_PROP_BUFFERSIZE, 1);

        int frame_width = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_WIDTH));
        int frame_height = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_HEIGHT));

        double fpsVideo = cap.get(cv::CAP_PROP_FPS);
        if (fpsVideo <= 0)
            fpsVideo = 15.0;
        int codec = {};
#ifndef WIN32
        codec = cv::VideoWriter::fourcc('a', 'v', 'c', '1');
#else
        codec = cv::VideoWriter::fourcc('m', 'p', '4', 'v');
#endif
        std::string filename_NA = "./ouput_NA.mp4"; // No annotations
        std::string filename_A = "./output_A.mp4";  // Annotated

        cv::VideoWriter writer;
        cv::VideoWriter annotatedWriter;
        writer.open(filename_NA, codec, fpsVideo, cv::Size(frame_width, frame_height), true);
        annotatedWriter.open(filename_A, codec, fpsVideo, cv::Size(frame_width, frame_height), true);

        if (!cap.isOpened())
        {
#ifndef WIN32
            rknn_destroy(ctx);
#endif
            return -1;
        }
        std::thread camThread(captureThread, std::ref(cap));

        int frame_skip = config.screen.frame_skip;

        cv::Mat frame;
        int frame_count = 0;
        int processed_count = 0;
        int detection_count = 0;

        long long total_preprocess_time = 0;
        long long total_inference_time = 0;
        long long total_postprocess_time = 0;

        bool paused = false;
        int waitTime = 1;
        auto prev_frame_time = Clock::now();
        std::vector<double> fps;

        while (true)
        {
            std::stringstream ss;
            if (pollForChanges())
            {
                cap.set(cv::CAP_PROP_BRIGHTNESS, config.camera.brightness);
                cap.set(cv::CAP_PROP_CONTRAST, config.camera.contrast);
                cap.set(cv::CAP_PROP_HUE, config.camera.hue);
                cap.set(cv::CAP_PROP_SATURATION, config.camera.saturation);
                cap.set(cv::CAP_PROP_GAIN, config.camera.gain);
                cap.set(cv::CAP_PROP_EXPOSURE, config.camera.exposure);
                cap.set(cv::CAP_PROP_WB_TEMPERATURE, config.camera.temperature);
                cap.set(cv::CAP_PROP_AUTO_WB, config.camera.temperature);
            }

            auto frame_start = Clock::now();
            prev_frame_time = frame_start;

#ifndef WIN32
            int INPUT_HEIGHT = config.yolo.input_dimensions;
            int INPUT_WIDTH = config.yolo.input_dimensions;
#else
            int INPUT_HEIGHT = config.yolo.input_dimensions;
            int INPUT_WIDTH = config.yolo.input_dimensions;
#endif

            float CONF_THRESHOLD = config.yolo.conf_threshold;
            float NMS_THRESHOLD = config.yolo.nms_threshold;
            auto preprocess_start = std::chrono::high_resolution_clock::now();

            if (!config.modes.video)
            {
                {
                    std::lock_guard<std::mutex> lock(frameMutex);
                    if (latestFrame.empty())
                        continue;
                    frame = latestFrame.clone();
                }
            }
            else
            {
                if (!cap.read(frame))
                    return -1;
            }

            if (config.camera.initial_blur != 0)
            {
                cv::GaussianBlur(frame, frame, cv::Size(config.camera.initial_blur, config.camera.initial_blur), 0);
            }

            int rotatedDegrees = config.screen.rotation;
            if (rotatedDegrees == 90)
                cv::rotate(frame, frame, cv::ROTATE_90_CLOCKWISE);
            if (rotatedDegrees == -90)
                cv::rotate(frame, frame, cv::ROTATE_90_COUNTERCLOCKWISE);
            if (rotatedDegrees == 180)
                cv::rotate(frame, frame, cv::ROTATE_180);

            frame_count++;

            if (frame_count % frame_skip != 0)
            {
                if (cv::waitKey(1) == 27)
                    break;
                continue;
            }
            processed_count++;

            if (config.modes.write_frame_to_file)
            {
                writer.write(frame);
            }

            cv::Mat resized;
            cv::resize(frame, resized, cv::Size(INPUT_WIDTH, INPUT_HEIGHT));
            cv::cvtColor(resized, resized, cv::COLOR_BGR2RGB);

            auto preprocess_end = std::chrono::high_resolution_clock::now();
            total_preprocess_time += std::chrono::duration_cast<std::chrono::milliseconds>(
                                         preprocess_end - preprocess_start)
                                         .count();

            auto inference_start = std::chrono::high_resolution_clock::now();

#ifndef WIN32
            rknn_input inputs[1];
            memset(inputs, 0, sizeof(inputs));
            inputs[0].index = 0;
            inputs[0].type = RKNN_TENSOR_UINT8;
            inputs[0].size = INPUT_WIDTH * INPUT_HEIGHT * 3;
            inputs[0].fmt = RKNN_TENSOR_NHWC;
            inputs[0].pass_through = 0;
            inputs[0].buf = resized.data;
            rknn_inputs_set(ctx, 1, inputs);

            rknn_run(ctx, nullptr);
            rknn_output outputs_rknn[1];
            memset(outputs_rknn, 0, sizeof(outputs_rknn));
            outputs_rknn[0].want_float = 1;
            rknn_outputs_get(ctx, 1, outputs_rknn, nullptr);
#else

            cv::Mat blob;
            cv::dnn::blobFromImage(frame, blob, 1.0 / 255.0, cv::Size(INPUT_WIDTH, INPUT_HEIGHT), cv::Scalar(0, 0, 0), true, false);

            net.setInput(blob);
            std::vector<cv::Mat> outputs;
            net.forward(outputs, net.getUnconnectedOutLayersNames());
#endif

            auto inference_end = std::chrono::high_resolution_clock::now();

            total_inference_time += std::chrono::duration_cast<std::chrono::milliseconds>(
                                        inference_end - inference_start)
                                        .count();

            auto postprocess_start = std::chrono::high_resolution_clock::now();

            int sizes[3] = {config.yolo.output_dimensions[0], config.yolo.output_dimensions[1], config.yolo.output_dimensions[2]};
            cv::Mat output_mat;
#ifndef WIN32
            cv::Mat output_mat_buf(3, sizes, CV_32F, outputs_rknn[0].buf);
            output_mat = output_mat_buf;
            std::vector outputs = {output_mat};
#endif

            std::vector<Detection> detections = ProcessYoloOutput(
                outputs, frame.cols, frame.rows,
                INPUT_WIDTH, INPUT_HEIGHT,
                CONF_THRESHOLD, NMS_THRESHOLD);

#ifndef WIN32
            rknn_outputs_release(ctx, 1, outputs_rknn);
#endif

            detection_count += static_cast<int>(detections.size());

            for (auto &det : detections)
            {
                cv::rectangle(frame, det.bounding_box, cv::FILLED);
            }

            detectionScheduler(frame, detections, config);

            int key = cv::waitKey(waitTime);

            auto postprocess_end = std::chrono::high_resolution_clock::now();
            total_postprocess_time += std::chrono::duration_cast<std::chrono::milliseconds>(
                                          postprocess_end - postprocess_start)
                                          .count();

            auto frame_end = Clock::now();
            using FrameDuration = std::chrono::duration<double>;
            auto delta = FrameDuration(frame_end - prev_frame_time).count();

            ss.clear();
            double sum = 0;
            for (int i = 0; i < fps.size(); i++)
            {
                sum += fps[i];
            }

            ss << std::fixed << std::setprecision(2) << "FPS: " << sum / fps.size();
            fps.emplace_back((1.f / delta));

            if (fps.size() >= 20)
            {
                fps.erase(fps.begin());
            }

            cap.set(cv::CAP_PROP_FPS, sum / fps.size());

            cv::putText(frame, ss.str(), cv::Point(10, 50),
                        cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(255, 0, 255), 2);

            if(sum/fps.size() <= 20) logger->warn("Deep stutter at " + std::to_string(sum/fps.size()) + "fps");

            if (config.modes.display) cv::imshow("detectEdgesBumper", frame);

            if (config.modes.write_frame_to_file)
            {
                annotatedWriter.write(frame);
            }

            if (key == 27)
            {
                cv::waitKey(10);
                capturing = false;
                break;
            }
            if (key == 112)
                paused = !paused;
            if (paused)
                waitTime = -1;
            else
                waitTime = 1;
        }
#ifndef WIN32
        rknn_destroy(ctx);
#endif
        cleanUp();

        spdlog::shutdown();

        if (!config.modes.video)
            camThread.join();

        cv::waitKey(500);

        cap.release();
        writer.release();

        logger->info("  Preprocess: " + std::to_string(total_preprocess_time / processed_count) + "ms\n");
        logger->info("  Inference: " + std::to_string(total_inference_time / processed_count) + "ms\n");
        logger->info("  Post-processing: " + std::to_string(total_postprocess_time / processed_count) + "ms\n");

        cv::destroyAllWindows();
        cv::waitKey(500);
    }
    catch (const cv::Exception &e)
    {
        logger->critical("OpenCV error" + std::string(e.what()) + "\n");
        return -1;
    }
    catch (const std::exception &e)
    {
        logger->critical("Error" + std::string(e.what()) + "\n");
        return -1;
    }
    catch (...)
    {
        logger->critical("Unknown error\n");
        return -1;
    }

    return 0;
}
