#include "analyze_detections.h"

#include <deque>
#include "log_to_file.h"
#include "config_extraction.h"
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/core/mat.hpp>
#include <sstream>
#include <fstream>
#include <ostream>
#include <string>
#include <tesseract/baseapi.h>
#include <chrono>
#include <vector>
#include <leptonica/allheaders.h>
#include <filesystem>
#include <future>
#include <iostream>
#include <unordered_set>
#include <opencv2/video/tracking.hpp>
#include "kalman_filter.h"
#include "top_down_view.h"
#include "thread_manager.h"

int levenshteinDist(const std::string &word1, const std::string &word2)
{
    const int size1 = static_cast<int>(word1.size());
    const int size2 = static_cast<int>(word2.size());
    std::vector verif(size1 + 1, std::vector<int>(size2 + 1));

    if (size1 == 0)
        return size2;
    if (size2 == 0)
        return size1;

    for (int i = 0; i <= size1; i++)
        verif[i][0] = i;
    for (int j = 0; j <= size2; j++)
        verif[0][j] = j;

    for (int i = 1; i <= size1; i++)
    {
        for (int j = 1; j <= size2; j++)
        {
            int cost = (word2[j - 1] == word1[i - 1]) ? 0 : 1;
            verif[i][j] = std::min(
                std::min(verif[i - 1][j] + 1, verif[i][j - 1] + 1),
                verif[i - 1][j - 1] + cost);
        }
    }

    return verif[size1][size2];
}

double getDistance(const double height, const Config &config)
{
    double focal_length_cm = config.focal_length;
    double known_height_cm = config.bumper_height;
    double pixel_height_cm = config.pixel_height;

    double distance;
    if (height > 0)
    {
        distance = (known_height_cm * focal_length_cm) / (height * pixel_height_cm);
        return distance;
    }
    return distance;
}

std::vector<double> getMeasurements(double distance, const Detection &detection, const Config &config,
                                    const std::vector<std::string> &visibleNumbers)
{
    thread_local std::unordered_map<std::string, kalmanFilter> filters;
    std::string label = detection.label;

    double SCREEN_WIDTH = config.screen_width;
    double SCREEN_HEIGHT = config.screen_height;
    double X_FOV = config.x_fov;
    double Y_FOV = config.y_fov;

    double max_cord_x = SCREEN_WIDTH / 2;
    double max_cord_y = SCREEN_HEIGHT / 2;

    double abs_bounding_x = detection.bounding_box.x + (0.5 * detection.bounding_box.width); // middle x
    double abs_bounding_y = detection.bounding_box.y + (detection.bounding_box.height);      // bottom y

    double offset_x = (abs_bounding_x - max_cord_x) / max_cord_x;
    double offset_y = (abs_bounding_y - max_cord_y) / max_cord_y;

    const double x_angle = (X_FOV / 2.0 * offset_x) * CV_PI / 180.0;
    const double y_angle = (Y_FOV / 2.0 * offset_y) * CV_PI / 180.0;

    const double x_coordinate = (distance / 100.0) * cos(y_angle) * cos(x_angle);
    const double y_coordinate = (distance / 100.0) * sin(x_angle);
    const double z_coordinate = (distance / 100.0) * sin(y_angle) * cos(x_angle);

    cv::Vec3d filtered;
    filtered[0] = x_coordinate;
    filtered[1] = y_coordinate;
    filtered[2] = z_coordinate;

    // Update tracker with robot position
    if (!label.empty())
    {
        auto it = filters.find(label);
        if (it == filters.end())
        {
            it = filters.emplace(label, config).first;
        }
        kalmanFilter &filter = it->second;
        filtered = filter.update(x_coordinate, y_coordinate, z_coordinate, static_cast<double>(1) / 5);
    }

    for (auto it = filters.begin(); it != filters.end();)
    {
        bool found = false;
        for (auto &num : visibleNumbers)
        {
            if (it->first == num)
            {
                found = true;
            }
        }
        if (!found)
        {
            it = filters.erase(it);
        }
        else
        {
            ++it;
        }
    }

    return {filtered[0], filtered[1], filtered[2]};
}

std::string getRobotLabel(Detection &det, const cv::Mat &hsv, const std::string teamNumbers[5], const Config &config,
                          bool cleanUp = false)
{
    thread_local std::unique_ptr<tesseract::TessBaseAPI> api;
    static thread_local bool init = false;

    if (cleanUp)
    {
        api->End();
        api.reset();
        return "-1";
    }

    if (!init)
    {
        api = std::make_unique<tesseract::TessBaseAPI>();
        api->Init("/usr/share/tessdata", "eng", tesseract::OEM_LSTM_ONLY);
        api->SetPageSegMode(tesseract::PSM_SINGLE_WORD);
        api->SetVariable("tessedit_char_whitelist", "0123456789");
        init = true;

        std::cout << "Tesseract initiated" << std::endl;
    }

    int ids = 0;

    cv::Mat img = hsv(det.bounding_box).clone();

    cv::Mat colorMask;
    cv::inRange(img, cv::Scalar(0, 0, 130), cv::Scalar(179, 170, 255), colorMask); // add config for these

    if (colorMask.cols < 300)
    {
        double scale = 300.0 / colorMask.cols;
        cv::resize(colorMask, colorMask, cv::Size(), scale, scale, cv::INTER_CUBIC);
    }

    cv::Mat final;
    cv::bitwise_not(colorMask, final);

    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3));
    cv::morphologyEx(final, final, cv::MORPH_OPEN, kernel);

    api->SetImage(final.data, final.cols, final.rows, 1, final.step);

    char *outText = api->GetUTF8Text();
    std::string result(outText);
    delete[] outText;

    result.erase(std::remove_if(result.begin(), result.end(), ::isspace), result.end());

    int minIndex = 0;

    int minDist = INT_MAX;
    if (!result.empty() && std::all_of(result.begin(), result.end(), ::isdigit))
    {
        for (int i = 0; i < 5; i++)
        {
            int d;
            d = levenshteinDist(result, teamNumbers[i]);
            if (d < minDist)
            {
                minDist = d;
                minIndex = i;
            }
        }
    }

    double distance = config.lev_distance;
    if (minDist > distance)
    {
        return det.id;
    }

    result = teamNumbers[minIndex];

    return result;
}

struct TrackedRobot
{
    int x = -1;
    int y = -1;
    std::string robot_id = "-1";
    int lostCounter = 0;
    std::string label;
    bool used = false;
};

std::vector<TrackedRobot> tracked(5);
std::vector<std::string> visibleNumbers;

struct OutputData
{
    double x;
    double y;
    double z;
    std::string label;
    Detection det;
    int id;
};

OutputData analyzeDetection(
    int id,
    std::string teamNumbers[5],
    cv::Mat &hsv,
    Detection det,
    const Config &config,
    bool cleanUp = false)
{
    if (cleanUp)
    {
        getRobotLabel(det, {}, {}, {}, true);
        return {};
    }

    auto t1 = std::chrono::high_resolution_clock::now();

    auto centerX = det.bounding_box.x + (0.5 * det.bounding_box.width);
    auto startCY = det.bounding_box.y + (0.5 * det.bounding_box.height);
    auto maxX = det.bounding_box.x + det.bounding_box.width;

    const cv::Vec3b *rowPtr = hsv.ptr<cv::Vec3b>(startCY);
    for (auto x = centerX; x < maxX; x++)
    {
        if (det.color != "red" && det.color != "blue")
        {
            auto color = rowPtr[(int)x];
            const double h = color[0];
            const double s = color[1];
            const double v = color[2];

            if (((h >= 0 && h <= 15) && (s >= 100 && s <= 255) && (v >= 130 && v <= 255)) ||
                ((h >= 170 && h <= 179) && (s >= 100 && s <= 255) && (v >= 130 && v <= 255)))
            {
                det.color = "red";
                break;
            }
            if ((h >= 80 && h <= 130) && (s >= 100 && s <= 255) && (v >= 80 && v <= 255))
            {
                det.color = "blue";
                break;
            }
            det.color = "";
        }
    }

    for (const auto &t : tracked)
    {
        if (t.robot_id == det.id)
        {
            if (t.label.empty())
                det.label = getRobotLabel(det, hsv, teamNumbers, config);
        }
    }

    double height = 0;
    double startHeight = 0;

    centerX = det.bounding_box.x + (0.5 * det.bounding_box.width);
    auto startTY = det.bounding_box.y;
    auto maxY = det.bounding_box.y + det.bounding_box.height;

    for (auto y = startTY; y < maxY; y++)
    {
        auto color = rowPtr[(int)y];
        const double h = color[0];
        const double s = color[1];
        const double v = color[2];

        if (((h >= 80 && h <= 120) && (s >= 100 && s <= 255) && (v >= 130 && v <= 255)) && det.color ==
                                                                                               "blue")
        {
            height++;
            if (startHeight == 0)
            {
                startHeight = y;
            }
        }
        else if ((((h >= 0 && h <= 15) && (s >= 100 && s <= 255) && (v >= 130 && v <= 255)) ||
                  ((h >= 170 && h <= 179) && (s >= 100 && s <= 255) && (v >= 130 && v <= 255))) &&
                 det.color == "red")
        {
            height++;
            if (startHeight == 0)
            {
                startHeight = y;
            }
        }
    }
    std::vector<double> measurements = getMeasurements(getDistance(height, config), det, config, visibleNumbers);

    OutputData data;
    data.x = measurements[0], data.y = measurements[1], data.z = measurements[2], data.label = det.label, data.det = det;
    auto t4 = std::chrono::high_resolution_clock::now();

    return data;
}

ThreadManager threadManager;

void detectionScheduler(std::string teamNumbers[5], cv::Mat &frame, std::vector<Detection> &detections,
                        const Config &config)
{
    if (detections.empty())
        return;

    visibleNumbers.clear();

    static std::deque<int> availableIDs;
    static bool idFilled = false;
    if (!idFilled)
    {
        for (int i = 0; i < 20; i++)
            availableIDs.push_back(i);
        idFilled = true;
    }

    for (auto &t : tracked)
        t.used = false;

    cv::Mat hsv;
    cv::cvtColor(frame, hsv, cv::COLOR_BGR2HSV);

    for (auto &det : detections)
    {
        int centerX = det.bounding_box.x + det.bounding_box.width / 2;
        int centerY = det.bounding_box.y + det.bounding_box.height / 2;

        TrackedRobot *bestMatch = nullptr;
        double minDist = std::numeric_limits<double>::max();

        for (auto &t : tracked)
        {
            if (t.used)
                continue;

            double dx = centerX - t.x;
            double dy = centerY - t.y;
            double dist = std::sqrt(dx * dx + dy * dy);

            if (dist < minDist && dist < config.maxDistanceThresholdX)
            {
                minDist = dist;
                bestMatch = &t;
            }
        }

        if (bestMatch)
        {
            bestMatch->x = centerX;
            bestMatch->y = centerY;
            bestMatch->lostCounter = 0;
            bestMatch->used = true;

            det.id = bestMatch->robot_id;
            if (!det.label.empty())
                bestMatch->label = det.label;
            std::stringstream ss;
            ss << "ID: " << det.id << " " << "Label: " << bestMatch->label;
            cv::putText(frame, ss.str(), cv::Point(det.bounding_box.x, det.bounding_box.y), cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(255, 255, 255), 2);
        }
        else if (!availableIDs.empty())
        {
            int newID = availableIDs.front();
            availableIDs.pop_front();

            TrackedRobot newRobot;
            newRobot.x = centerX;
            newRobot.y = centerY;
            newRobot.robot_id = std::to_string(newID);
            newRobot.lostCounter = 0;
            newRobot.used = true;

            tracked.push_back(newRobot);
            det.id = newRobot.robot_id;
        }
        else
        {
            det.id.clear();
        }
    }

    for (auto it = tracked.begin(); it != tracked.end();)
    {
        if (!it->used)
            it->lostCounter++;

        if (it->lostCounter >= 10)
        {
            if (it->robot_id != "-1")
            {
                availableIDs.push_back(std::stoi(it->robot_id));
            }
            it = tracked.erase(it);
        }
        else
            ++it;
    }

    std::vector<std::future<OutputData>> futures;

    for (size_t i = 0; i < detections.size(); i++)
    {
        futures.push_back(threadManager.enqueue([&hsv, teamNumbers, config, i, &detections]()
                                                {
                try {
                    return analyzeDetection(i, teamNumbers, hsv, detections[i], config);
                } catch (...) {
                    return OutputData{};
                } }));
    }

    std::vector<OutputData> results;
    results.reserve(futures.size());

    for (auto &fut : futures)
    {
        results.push_back(fut.get());
    }

    for (size_t i = 0; i < detections.size(); i++)
    {
        detections[i].label = results[i].label;
    }

    for (auto &det : detections)
    {
        for (auto &result : results)
        {
            cv::line(frame, cv::Point(640, 720), 
                cv::Point(result.det.bounding_box.x + result.det.bounding_box.width / 2, result.det.bounding_box.y + result.det.bounding_box.height / 2), 
                result.det.color == "blue" ? cv::Scalar(255, 0, 0) : cv::Scalar(0, 0, 255), 1);
        }

        for (auto &t : tracked)
        {
            if (det.id == t.robot_id && !det.label.empty())
            {
                t.label = det.label;
                std::cout << det.label << std::endl;
            }
        }
    }
}

void cleanUp()
{
    std::vector<std::future<void>> cleanupFutures;

    for (int i = 0; i < threadManager.numThreads; ++i)
    {
        cleanupFutures.push_back(threadManager.enqueue([]()
                                                       {
            std::string teamNumbers[5] = {"0", "0", "0", "0", "0"};
            cv::Mat emptyHsv;
            Detection emptyDet;
            Config defaultConfig;
            getRobotLabel(emptyDet, emptyHsv, teamNumbers, defaultConfig, true); }));
    }

    for (auto &fut : cleanupFutures)
    {
        fut.get();
    }
    threadManager.shutdown();
}
