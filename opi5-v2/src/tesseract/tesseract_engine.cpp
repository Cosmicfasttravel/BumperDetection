#include "tesseract_engine.h"

#include <tesseract/baseapi.h>
#include <leptonica/allheaders.h>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

#include "core/config/config_extraction.h"
#include "core/detection/structure/detection_data.h"
#include "global/internal/built_components.h"
#include "log/logger.h"
#include "spdlog/logger.h"

std::atomic<int> TesseractEngine::ocrCounter{0};
std::mutex TesseractEngine::ocrMutex;

TesseractEngine::TesseractEngine() = default;

TesseractEngine::~TesseractEngine() {
    cleanTesseractEngine();
}

tesseract::TessBaseAPI &TesseractEngine::getTesseract() {
    static thread_local tesseract::TessBaseAPI api;
    return api;
}

TesseractEngine& TesseractEngine::current() {
    static thread_local TesseractEngine engine;
    return engine;
}

bool TesseractEngine::initTesseractEngine(const Config &config) {
    try {
        static thread_local std::atomic<bool> init = false;

        if (init) return true;

        auto &api = getTesseract();

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
    } catch (...) {
        logging::write("Error occurred with initiation, retrying...", spdlog::level::err);
        return false;
    }
    return true;
}

cv::Mat TesseractEngine::processImage(const Config &config, const cv::Mat &hsvImage, Detection detection) {
    static cv::Mat emptyMat;

    cv::Rect safeBB = detection.boundingBox & cv::Rect(0, 0, hsvImage.cols, hsvImage.rows);
    if (safeBB.empty()) {
        return emptyMat;
    }
    cv::Mat hsv = hsvImage(safeBB);

    cv::Mat colorMask;

    cv::inRange(hsv,
        cv::Scalar(config.ocr.mask_thresholds.hue_lower, config.ocr.mask_thresholds.saturation_lower,
    config.ocr.mask_thresholds.value_lower),
    cv::Scalar(config.ocr.mask_thresholds.hue_upper, config.ocr.mask_thresholds.saturation_upper,
    config.ocr.mask_thresholds.value_upper), colorMask);

    if (colorMask.cols < config.ocr.min_img_size) {
        double scale = static_cast<float>(config.ocr.min_img_size) / static_cast<float>(colorMask.cols);
        cv::resize(colorMask, colorMask, cv::Size(), scale, scale, cv::INTER_CUBIC);
    }

    cv::Mat final = colorMask;

    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT,
                                               cv::Size(config.ocr.morphology_kernel_size,
                                                        config.ocr.morphology_kernel_size));

    // cv::imwrite("../src/tesseract/img/ocr_debug" + std::to_string(std::rand()) + ".png", final);

    if ((final.empty() || final.cols <= 0 || final.rows <= 0)) {
        logging::write("Final was empty");
        return emptyMat;
    }

    return final;
}

std::string TesseractEngine::extractText(const Config &config, cv::Mat &img) {
    try {
        {
            std::lock_guard l(ocrMutex);
            if (ocrCounter >= config.thread_pool_size) return "-1";
            ++ocrCounter;
        }

        thread_local auto &api = getTesseract();

        api.SetImage(img.data, img.cols, img.rows, 1, img.step);
        api.SetSourceResolution(70);
        char *outText = api.GetUTF8Text();

        if (outText == nullptr) {
            logging::write("Out Text from tess was null");
            return "";
        }

        std::string result(outText);
        delete[] outText;

        std::erase_if(result, ::isspace); {
            std::lock_guard l(ocrMutex);
            --ocrCounter;
        }

        return result;
    } catch (std::exception &e) {
        {
            std::lock_guard l(ocrMutex);
            --ocrCounter;
        }

        logging::write(e.what(), spdlog::level::err);
        return "";
    } catch (...) {
        {
            std::lock_guard l(ocrMutex);
            --ocrCounter;
        }

        logging::write("Other issue occurred", spdlog::level::err);
        return "";
    }
}

void TesseractEngine::cleanTesseractEngine() {
    try {
        thread_local auto &api = getTesseract();

        api.End();
    } catch (...) {
        logging::write("Error occurred with cleanup", spdlog::level::err);
    }
}

int TesseractEngine::levDistance(const std::string &s1, const std::string &s2) {
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

std::string TesseractEngine::findMinDistance(const Config &config, std::string text, Detection detection) {
    int minIndex = 0;

    int minDist = INT_MAX;
    if (!text.empty() && std::ranges::all_of(text, ::isdigit)) {
        for (int i = 0; i < 3; i++) {
            int d = {};
            if (detection.color == Color::BLUE) d = levDistance(text, config.teams.blueTeams[i]);
            if (detection.color == Color::RED) d = levDistance(text, config.teams.redTeams[i]);
            else logging::write("Not a valid color", spdlog::level::err);
            if (d < minDist) {
                minDist = d;
                minIndex = i;
            }
        }
    }

    if (detection.color == Color::BLUE) text = config.teams.blueTeams[minIndex];
    if (detection.color == Color::RED) text = config.teams.redTeams[minIndex];

    if (minDist > config.ocr.lev_distance) {
        logging::write("Lev distance larger than minimum distance");
        return "";
    }

    return text;
}

std::string TesseractEngine::tesseractEngine(const cv::Mat &img, Detection detection) {
    thread_local Config config = config::getLatestCopy();
    if (config::checkConfigVersion(config)) config = config::getLatestCopy();

    if (!initTesseractEngine(config)) {
        logging::write("Tess init failed", spdlog::level::err);
        return "-1";
    }

    cv::Mat finalImg = processImage(config, img, detection);

    std::string result;
    if (!finalImg.empty()) result = extractText(config, finalImg);
    if (!result.empty()) result = findMinDistance(config, result, detection);

    return result;
}
