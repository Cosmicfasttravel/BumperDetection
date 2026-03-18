#include "analyze_detections.h"

#include <deque>
#include "config_extraction.h"
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/core/mat.hpp>
#include <sstream>
#include <fstream>
#include <ostream>
#include <tesseract/baseapi.h>
#include <chrono>
#include <vector>
#include <leptonica/allheaders.h>
#include <filesystem>
#include <iostream>
#include <unordered_set>
#include <opencv2/video/tracking.hpp>
#include "kalman_filter.h"
#include "top_down_view.h"

struct Position3D
{
    double z_cm;
};

void logTimes(const std::vector<std::chrono::system_clock::time_point> &durations)
{
    std::stringstream ss;
    if (durations.size() < 1)
    {
        return;
    }
    ss << "Timings:\n";
    for (size_t i = 1; i < durations.size(); i++)
    {
        if (i != 1)
        {
            ss << '\n';
        }
        ss << "\tt" << i << "-t" << i - 1 << "\t"
           << std::chrono::duration_cast<std::chrono::milliseconds>(durations.at(i) - durations.at(i - 1)).count()
           << "ms";
    }
    std::cout << ss.str() << std::endl;
}

// Global tracker instance
static RobotTracker g_tracker;

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

template <typename T>
T findMode(const std::vector<T> &data)
{
    std::unordered_map<T, int> counts;
    for (const T &value : data)
    {
        ++counts[value];
    }

    int maxCount = 0;
    T mostFrequentVal = T();

    for (const auto &pair : counts)
    {
        if (pair.second > maxCount)
        {
            maxCount = pair.second;
            mostFrequentVal = pair.first;
        }
    }

    return mostFrequentVal;
}

Position3D getPosition3D(
    const double height, const Config &config)
{
    double focal_length_cm = config.focal_length;
    double known_height_cm = config.bumper_height;
    double pixel_height_cm = config.pixel_height;

    Position3D pos{};
    if (height > 0)
    {
        pos.z_cm = (known_height_cm * focal_length_cm) / (height * pixel_height_cm);
        return pos;
    }
    return pos;
}

void drawMeasurements(
    Position3D &pos,
    const Detection &detection, const Config &config, const std::vector<std::string> &visibleNumbers)
{
    static std::unordered_map<std::string, kalmanFilter> filters;
    std::string label = detection.label;

    double SCREEN_WIDTH = config.screen_width;
    double SCREEN_HEIGHT = config.screen_height;
    double X_FOV = config.x_fov;
    double Y_FOV = config.y_fov;

    static const double max_cord_x = SCREEN_WIDTH / 2;
    static const double max_cord_y = SCREEN_HEIGHT / 2;

    double abs_bounding_x = detection.bounding_box.x + (0.5 * detection.bounding_box.width);
    double abs_bounding_y = detection.bounding_box.y + (0.5 * detection.bounding_box.height);

    double offset_x = (abs_bounding_x - max_cord_x) / max_cord_x;
    double offset_y = (abs_bounding_y - max_cord_y) / max_cord_y;

    const double x_angle = (X_FOV / 2.0 * offset_x) * CV_PI / 180.0;
    const double y_angle = (Y_FOV / 2.0 * offset_y) * CV_PI / 180.0;

    double forward = pos.z_cm / cos(x_angle);
    const double x_coordinate = forward / 100;
    const double y_coordinate = forward * tan(x_angle) / 100;
    const double z_coordinate = pos.z_cm * tan(y_angle) / 100;

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
    if (detection.color == "red")
    {
        g_tracker.updateRobotPosition(filtered[0], filtered[1], filtered[2], label, cv::Scalar(0, 0, 255));
    }
    else if (detection.color == "blue")
    {
        g_tracker.updateRobotPosition(filtered[0], filtered[1], filtered[2], label, cv::Scalar(255, 0, 0));
    }
    else
    {
        g_tracker.updateRobotPosition(filtered[0], filtered[1], filtered[2], label, cv::Scalar(0, 0, 0));
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

    if (config.loggingMode == 1)
    {
        std::ofstream outFile("../log.txt", std::ios_base::app);
        outFile << "x: " << filtered[0] << "\n";
        outFile << "y: " << filtered[1] << "\n";
        outFile << "z: " << filtered[2] << "\n\n";
        outFile.close();
    }
}

void startOCR()
{
    api = new tesseract::TessBaseAPI();
    api->Init("/usr/share/tessdata", "eng", tesseract::OEM_LSTM_ONLY);

    api->SetPageSegMode(tesseract::PSM_SINGLE_WORD);
    api->SetVariable("tessedit_char_whitelist", "0123456789");
}

void endOCR()
{
    api->End();
    delete api;
}

std::vector<std::string> findNumbers(std::vector<Detection> &detections, const cv::Mat &hsvFrame,
                                     const std::string teamNumbers[5], const Config &config)
{
    std::vector<std::string> visibleNumbers;
    for (auto &det : detections)
    {
        cv::Mat img = hsvFrame(det.bounding_box).clone();

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
                std::cout << teamNumbers[i] << std::endl;
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
            continue;
        }

        result = teamNumbers[minIndex];

        det.label = result;
        visibleNumbers.emplace_back(result);
    }
    return visibleNumbers;
}

void analyzeDetections(
    const std::string teamNumbers[5],
    cv::Mat &frame,
    std::vector<Detection> &detections,
    const Config &config)
{
    // Clear previous robot positions
    g_tracker.clearRobots();
    if (!detections.empty())
    {
        cv::Mat hsv;

        cv::cvtColor(frame, hsv, cv::COLOR_BGR2HSV);

        const std::vector<std::string> visibleNumbers = findNumbers(detections, hsv, teamNumbers, config);
        for (int i = 0; i < detections.size(); i++)
        {

            if (detections[i].label.empty())
            {
                detections[i].label = "robot_";
                detections[i].label += std::to_string(i);
            }
        }
        auto t1 = std::chrono::high_resolution_clock::now();
        for (auto &det : detections)
        {
            double height = 0;
            double startHeight = 0;

            auto centerX = det.bounding_box.x + (0.5 * det.bounding_box.width);
            auto startCY = det.bounding_box.y + (0.5 * det.bounding_box.height);
            auto startTY = det.bounding_box.y;
            auto startX = det.bounding_box.x;
            auto maxY = det.bounding_box.y + det.bounding_box.height;
            auto maxX = det.bounding_box.x + det.bounding_box.width;

            // walks right finding red or blue pixels
            for (auto x = centerX; x < maxX; x++)
            {
                if (det.color != "red" && det.color != "blue")
                {
                    auto color = hsv.at<cv::Vec3b>(startCY, x);
                    const double h = color[0];
                    const double s = color[1];
                    const double v = color[2];

                    if (((h >= 0 && h <= 15) && (s >= 100 && s <= 255) && (v >= 130 && v <= 255)) ||
                        ((h >= 170 && h <= 179) && (s >= 100 && s <= 255) && (v >= 130 && v <= 255)))
                    {
                        det.color = "red";
                        break;
                    }
                    if ((h >= 80 && h <= 120) && (s >= 100 && s <= 255) && (v >= 130 && v <= 255))
                    {
                        det.color = "blue";
                        break;
                    }
                }
            }

            int maxColHeight = 0;
            for (auto x = startX; x < maxX; x++)
            {
                int colHeight = 0;
                for (auto y = startTY; y < maxY; y++)
                {
                    auto color = hsv.at<cv::Vec3b>(y, x);
                    const double h = color[0];
                    const double s = color[1];
                    const double v = color[2];

                    if (((h >= 80 && h <= 120) && (s >= 100 && s <= 255) && (v >= 130 && v <= 255)) && det.color == "blue")
                    {
                        colHeight++;
                    }
                    else if ((((h >= 0 && h <= 15) && (s >= 100 && s <= 255) && (v >= 130 && v <= 255)) ||
                              ((h >= 170 && h <= 179) && (s >= 100 && s <= 255) && (v >= 130 && v <= 255))) &&
                             det.color == "red")
                    {
                        colHeight++;
                    }
                }
                maxColHeight = std::max(maxColHeight, colHeight);
            }
            height = maxColHeight;
            Position3D pos = getPosition3D(height, config);
            drawMeasurements(pos, det, config, visibleNumbers);
        }
    }

    auto t2 = std::chrono::high_resolution_clock::now();

    // Render top-down view
    g_tracker.render();
    cv::imshow("detectEdgesBumper", frame);
}
