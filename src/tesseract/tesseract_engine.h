#ifndef BUMPERDETECTION_TESSERACT_ENGINE_H
#define BUMPERDETECTION_TESSERACT_ENGINE_H
#include <atomic>
#include <tesseract/baseapi.h>
#include <leptonica/allheaders.h>
#include <opencv2/core/mat.hpp>

#include "core/config/config_types.h"

class tesseract_engine {
public:

    tesseract_engine();
    ~tesseract_engine();

    std::string tesseractEngine(cv::Mat &img);

private:
    static Config config;
    static std::atomic<int> ocrCounter;

    bool initTesseractEngine();

    cv::Mat processImage(const cv::Mat &hsvImage);

    tesseract::TessBaseAPI& getTesseract();

    std::string extractText(cv::Mat &img);
    std::string findMinDistance(std::string text);

    static int levDistance(const std::string &s1, const std::string &s2);
    static void updateConfig();

    void cleanTesseractEngine();
};


#endif //BUMPERDETECTION_TESSERACT_ENGINE_H
