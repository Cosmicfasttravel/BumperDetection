#ifndef BUMPERDETECTION_TESSERACT_ENGINE_H
#define BUMPERDETECTION_TESSERACT_ENGINE_H
#include <atomic>
#include <mutex>
#include <tesseract/baseapi.h>
#include <leptonica/allheaders.h>
#include <opencv2/core/mat.hpp>

#include "core/config/config_types.h"

class tesseract_engine {
public:

    tesseract_engine();
    ~tesseract_engine();

    std::string tesseractEngine(const cv::Mat &img);

private:
    static std::atomic<int> ocrCounter;
    static std::mutex ocrMutex;

    bool initTesseractEngine(const Config &config);

    cv::Mat processImage(const Config &config ,const cv::Mat &hsvImage);

    tesseract::TessBaseAPI& getTesseract();

    std::string extractText(const Config &config, cv::Mat &img);
    std::string findMinDistance(const Config &config, std::string text);

    static int levDistance(const std::string &s1, const std::string &s2);

    void cleanTesseractEngine();
};


#endif //BUMPERDETECTION_TESSERACT_ENGINE_H
