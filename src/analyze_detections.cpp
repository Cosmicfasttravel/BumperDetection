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

std::vector<double> getMeasurements(double distance, const Detection &detection, const Config &config, const std::vector<std::string> &visibleNumbers)
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

std::string getRobotLabel(Detection &det, const cv::Mat &hsv, const std::string teamNumbers[5], const Config &config, bool cleanUp = false)
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
        api->SetVariable("parallel_tess", "0");
        init = true;

        std::cout << "Tesseract initiated" << std::endl;
    }

    if (det.color.empty())
        return "";

    int ids = 0;

    cv::Mat img = hsv(det.bounding_box).clone();

    cv::Mat colorMask;
    cv::inRange(img, cv::Scalar(0, 0, 200), cv::Scalar(179, 70, 255), colorMask);

    cv::Mat gray;
    cv::cvtColor(img, gray, cv::COLOR_HSV2BGR);
    cv::cvtColor(gray, gray, cv::COLOR_BGR2GRAY);

    if (gray.cols < 300)
    {
        double scale = 300.0 / gray.cols;
        cv::resize(gray, gray, cv::Size(), scale, scale, cv::INTER_CUBIC);
        cv::resize(colorMask, colorMask, cv::Size(), scale, scale, cv::INTER_CUBIC);
    }

    cv::Mat final;
    cv::bitwise_not(colorMask, final);

    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3));
    cv::morphologyEx(final, final, cv::MORPH_CLOSE, kernel);
    cv::morphologyEx(final, final, cv::MORPH_OPEN, kernel);

    api->Clear();
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
        return {};
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
std::deque<int> availableIDs = {0, 1, 2, 3, 4};
std::vector<std::string> visibleNumbers;

std::vector<double> analyzeDetection(
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

    det.label = getRobotLabel(det, hsv, teamNumbers, config);

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
            else if ((h >= 80 && h <= 120) && (s >= 100 && s <= 255) && (v >= 130 && v <= 255))
            {
                det.color = "blue";
                break;
            }
            else
            {
                det.color = "";
            }
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

    std::vector<double> data = {measurements[0], measurements[1], measurements[2], (det.label.empty()) ? 0 : std::stod(det.label)};
    auto t4 = std::chrono::high_resolution_clock::now();

    return data;
}

ThreadManager threadManager;
void detectionScheduler(std::string teamNumbers[5], cv::Mat &frame, std::vector<Detection> &detections, const Config &config)
{
    if (detections.empty())
        return;

    visibleNumbers.clear();

    cv::Mat hsv;
    cv::cvtColor(frame, hsv, cv::COLOR_BGR2HSV);

    std::vector<std::future<std::vector<double>>> futures;
    for (auto &det : detections)
    {
        if (availableIDs.empty())
        {
            for (int i = 0; i < 5; i++)
            {
                availableIDs.emplace_back(i);
            }
        }

        int centerY = det.bounding_box.y + det.bounding_box.height / 2;

        TrackedRobot *bestMatch = nullptr;
        double minDist = std::numeric_limits<double>::max();

        int bestMatchIndex = -1;
        auto centerX = det.bounding_box.x + (0.5 * det.bounding_box.width);
        for (size_t i = 0; i < tracked.size(); i++)
        {
            if (tracked[i].used)
                continue;
            double dx = centerX - tracked[i].x;
            double dy = centerY - tracked[i].y;
            double dist = std::sqrt(dx * dx + dy * dy);

            if (dist < minDist && dist < config.maxDistanceThresholdX)
            {
                minDist = dist;
                bestMatchIndex = i;
            }
        }
        auto t2 = std::chrono::high_resolution_clock::now();
        if (bestMatchIndex != -1)
        {
            if (tracked[bestMatchIndex].label.empty())
            {
                tracked[bestMatchIndex].used = true;
            }

            tracked[bestMatchIndex].x = centerX;
            tracked[bestMatchIndex].y = centerY;
            tracked[bestMatchIndex].lostCounter = 0;
            det.id = tracked[bestMatchIndex].robot_id;

            if (!det.label.empty())
                tracked[bestMatchIndex].label = det.label;
        }
        else
        {
            if (availableIDs.empty())
            {
                std::cout << "WARNING: No available IDs!" << std::endl;
            }

            int newID = availableIDs.front();
            availableIDs.pop_front();

            TrackedRobot newRobot;
            newRobot.x = centerX;
            newRobot.y = centerY;
            newRobot.robot_id = std::to_string(newID);
            newRobot.lostCounter = 0;
            tracked.emplace_back(newRobot);

            det.id = newRobot.robot_id;
        }
        auto t3 = std::chrono::high_resolution_clock::now();

        for (size_t i = 0; i < tracked.size(); i++)
        {
            bool matched = false;

            if (det.id == tracked[i].robot_id)
            {
                matched = true;
                if (!det.label.empty())
                    tracked[i].label = det.label;
                break;
            }

            if (!matched)
            {
                tracked[i].lostCounter++;
                if (tracked[i].lostCounter >= 10)
                {
                    tracked[i].x = -1;
                    tracked[i].y = -1;
                    tracked[i].lostCounter = 0;
                    availableIDs.push_back(std::stoi(tracked[i].robot_id));
                }
            }
        }

        if (tracked.size() > 10)
            tracked.erase(tracked.begin());

        for (const auto &t : tracked)
        {
            if (!t.label.empty())
            {
                visibleNumbers.emplace_back(t.label);
            }
        }
    }
    for (auto det : detections)
    {
        futures.push_back(threadManager.enqueue([&hsv, teamNumbers, config, det]() mutable
                                                { try { return analyzeDetection(teamNumbers, hsv, det, config);
    } catch (const std::exception &e) {
        std::cerr << "Exception in thread: " << e.what() << std::endl;
        return std::vector<double>{0,0,0,0,0};
    } catch (...) {
        std::cerr << "Unknown exception in thread!" << std::endl;
        return std::vector<double>{0,0,0,0,0};
    } }));
    }

    std::vector<std::vector<double>> results;
    results.reserve(futures.size());

    for (auto &fut : futures)
    {
        results.push_back(fut.get());
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