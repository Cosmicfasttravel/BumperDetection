#include "detector.h"
#include "process_detection.h"
#include "core/config/config_extraction.h"
#include "log/logger.h"
#include <iostream>
#include "global/build_info.h"


Detector::Detector() : initialized(false), INPUT_HEIGHT(0), INPUT_WIDTH(0), NMS_THRESHOLD(0), CONF_THRESHOLD(0) {
}

Detector::~Detector() {
    if (initialized) {
#ifdef __aarch64__
        rknn_destroy(ctx);
#endif
    }
}

void Detector::initializeDetector() {
    static Config config = config::getLatestCopy();
    if (config::checkConfigVersion(config)) config = config::getLatestCopy();

#ifdef __aarch64__
    std::string model_path = config.input_paths.rknn_path;
    FILE *fp = fopen(model_path.c_str(), "rb");
    if (!fp) {
        logging::write("Failed to open model", spdlog::level::err);
        return;
    }
    fseek(fp, 0, SEEK_END);
    long model_size = ftell(fp);
    rewind(fp);
    void *model_data = malloc(model_size);
    fread(model_data, 1, model_size, fp);
    fclose(fp);

    int ret = rknn_init(&ctx, model_data, model_size, 0, nullptr);

    free(model_data);
    if (ret != 0) {
        logging::write("Failed to init RKNN", spdlog::level::warn);
        return;
    }
    rknn_core_mask core_mask = RKNN_NPU_CORE_0_1_2;
    rknn_set_core_mask(ctx, core_mask);

    logging::write("RKNN Loaded sucessfully");
    initialized = true;
#else
    if (build_info::is_cpu) {
        std::string model_path = config.input_paths.onnx_path;
        net = cv::dnn::readNetFromONNX(model_path);
        if (net.empty()) {
            return;
        }
        initialized = true;
        logging::write("ONNX Loaded sucessfully");
    }
    INPUT_HEIGHT = config.yolo.input_dimensions;
    INPUT_WIDTH = config.yolo.input_dimensions;

    CONF_THRESHOLD = static_cast<float>(config.yolo.conf_threshold);
    NMS_THRESHOLD = static_cast<float>(config.yolo.nms_threshold);
#endif
}

std::vector<Detection> Detector::detect(const cv::Mat &img) {
    static Config config = config::getLatestCopy();
    if (config::checkConfigVersion(config)) config = config::getLatestCopy();

    if (!initialized) {
        return {};
    }

    std::vector<Detection> detections;

#ifdef __aarch64__
    cv::Mat resized;
    cv::resize(img, resized, cv::Size(INPUT_WIDTH, INPUT_HEIGHT));
    cv::cvtColor(resized, resized, cv::COLOR_BGR2RGB);

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

    cv::Mat outputMat;
    int sizes[3] = {
        config.yolo.output_dimensions[0], config.yolo.output_dimensions[1], config.yolo.output_dimensions[2]
    };
    cv::Mat output_mat_buf(3, sizes, CV_32F, outputs_rknn[0].buf);
    outputMat = output_mat_buf.clone();
    std::vector outputs = {outputMat};

    detections = process_detection::ProcessYoloOutput(
        outputs, img.cols, img.rows,
        INPUT_WIDTH, INPUT_HEIGHT,
        CONF_THRESHOLD, NMS_THRESHOLD);

    rknn_outputs_release(ctx, 1, outputs_rknn);
#else
    if (build_info::is_cpu) {
        cv::Mat blob;
        cv::dnn::blobFromImage(img, blob, 1.0 / 255.0, cv::Size(INPUT_WIDTH, INPUT_HEIGHT), cv::Scalar(0, 0, 0), true,
                               false);

        net.setInput(blob);
        std::vector<cv::Mat> outputs;
        net.forward(outputs, net.getUnconnectedOutLayersNames());

        detections = process_detection::ProcessYoloOutput(
            outputs, img.cols, img.rows,
            INPUT_WIDTH, INPUT_HEIGHT,
            CONF_THRESHOLD, NMS_THRESHOLD);
    }
#endif
    return detections;
}
