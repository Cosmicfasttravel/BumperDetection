#include "tesseract_engine.h"

#include <tesseract/baseapi.h>
#include <leptonica/allheaders.h>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

#include "core/config/config_extraction.h"
#include "global/internal/built_components.h"
#include "log/logger.h"
#include "spdlog/logger.h"

tesseract_engine::tesseract_engine() {
    config = config::getLatestCopy();
}

tesseract_engine::~tesseract_engine() {

}

tesseract::TessBaseAPI& tesseract_engine::getTesseract() {
    static thread_local tesseract::TessBaseAPI api;
    return api;
}

bool tesseract_engine::initTesseractEngine() {
    try {
        static std::atomic<bool> init = false;

        updateConfig();

        if (init) return true;

        thread_local auto& api = getTesseract();

        if (config.ocr.mode == "default" || config.ocr.mode == "tessonly")
            api.Init(
                config.ocr.tessdata_path.c_str(), "eng", tesseract::OEM_TESSERACT_ONLY);
        if (config.ocr.mode == "lstmonly")
            api.Init(config.ocr.tessdata_path.c_str(), "eng",
                      tesseract::OEM_LSTM_ONLY);
        if (config.ocr.mode == "combined")
            api.Init(config.ocr.tessdata_path.c_str(), "eng",
                      tesseract::OEM_TESSERACT_LSTM_COMBINED);

        api.SetPageSegMode(tesseract::PSM_SINGLE_WORD);
        api.SetVariable("tessedit_char_whitelist", "0123456789");

        init = true;
    }
    catch (...) {
        logging::write("Error occurred with initiation, retrying...", spdlog::level::err);
        return false;
    }
    return true;
}

cv::Mat &tesseract_engine::proccessImage(const cv::Mat &hsvImage) {
    // When the data structure for the detection is created ensure that these lines are performed (saves large performance)
    // cv::Rect safeBB = det.bounding_box & cv::Rect(0, 0, hsvImage.cols, hsvImage.rows);
    // if (safeBB.empty()) {
    //     --ocrCounter;
    //     return {};
    // }

    cv::Mat colorMask;

    cv::inRange(
            hsvImage, cv::Scalar(config.ocr.mask_thresholds.hue_lower, config.ocr.mask_thresholds.saturation_lower,
                            config.ocr.mask_thresholds.value_lower),
            cv::Scalar(config.ocr.mask_thresholds.hue_upper, config.ocr.mask_thresholds.saturation_upper,
                       config.ocr.mask_thresholds.value_upper), colorMask);

    if (colorMask.cols < config.ocr.min_img_size) {
        double scale = static_cast<float>(config.ocr.min_img_size) / static_cast<float>(colorMask.cols);
        cv::resize(colorMask, colorMask, cv::Size(), scale, scale, cv::INTER_CUBIC);
    }

    cv::Mat final;
    cv::bitwise_not(colorMask, final);

    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT,
                                               cv::Size(config.ocr.morphology_kernel_size,
                                                        config.ocr.morphology_kernel_size));
    cv::morphologyEx(final, final, cv::MORPH_OPEN, kernel);

    if (final.empty() || final.cols <= 0 || final.rows <= 0) {
        --ocrCounter;
        return {};
    }

    return final;
}

std::string tesseract_engine::extractText(cv::Mat &img) {
    thread_local auto& api = getTesseract();

    api.SetImage(img.data, img.cols, img.rows, 1, img.step);
    api.SetSourceResolution(70);
    char *outText = api.GetUTF8Text();
    std::string result(outText);
    delete[] outText;

    std::erase_if(result, ::isspace);

    return result;
}

void tesseract_engine::cleanTesseractEngine() {
    try {
        thread_local auto& api = getTesseract();

        api.End();
    }
    catch (...) {
        logging::write("Error occurred with cleanup", spdlog::level::err);
    }
}

void tesseract_engine::updateConfig() {
    if (config::checkConfigVersion(config)) config = config::getLatestCopy();
}

int tesseract_engine::levDistance(const std::string &s1, const std::string &s2) {
    const int size1 = static_cast<int>(s1.size());
    const int size2 = static_cast<int>(s2.size());
    std::vector verif(size1 + 1, std::vector<int>(size2 + 1));

    if (size1 == 0)
        return size2;
    if (size2 == 0)
        return size1;

    for (int i = 0; i <= size1; i++)
        verif[i][0] = i;
    for (int j = 0; j <= size2; j++)
        verif[0][j] = j;

    for (int i = 1; i <= size1; i++) {
        for (int j = 1; j <= size2; j++) {
            int cost = (s2[j - 1] == s1[i - 1]) ? 0 : 1;
            verif[i][j] = std::min(
                std::min(verif[i - 1][j] + 1, verif[i][j - 1] + 1),
                verif[i - 1][j - 1] + cost);
        }
    }

    return verif[size1][size2];
}

std::string tesseract_engine::findMinDistance(std::string text) {
    int minIndex = 0;

    int minDist = INT_MAX;
    if (!text.empty() && std::ranges::all_of(text, ::isdigit)) {
        for (int i = 0; i < 3; i++) {
            int d = {};
            // Needs correct detection data structure
            // if (det.color == "blue") d = levDistance(text, config.teams.blueTeams[i]);
            // if (det.color == "red") d = levDistance(text, config.teams.redTeams[i]);

            if (d < minDist) {
                minDist = d;
                minIndex = i;
            }
        }
    }
    // Needs correct detection data structure
    // if (det.color == "blue") text = config.teams.blueTeams[minIndex];
    // if (det.color == "red") text = config.teams.redTeams[minIndex];

    if (minDist > config.ocr.lev_distance) {
        return "";
    }

    return text;
}

std::string tesseract_engine::tesseractEngine(cv::Mat &img) { // Needs detection data structure which is passed to the helper internal functions
    if (!initTesseractEngine()) {
        return "";
    }
    cv::Mat finalImg = proccessImage(img); // Needs detection
    std::string result = extractText(finalImg);
    result = findMinDistance(result);

    return result;
}

