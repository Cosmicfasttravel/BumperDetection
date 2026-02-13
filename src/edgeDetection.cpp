#include "edgeDetection.h"

#include <deque>
#include <iostream>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/core/mat.hpp>
#include <sstream>
#include <tesseract/baseapi.h>
#include <leptonica/allheaders.h>
#include <filesystem>
#include <iomanip>
#include <opencv2/video/tracking.hpp>
#include "kalmanFilter.h"
#include "topDownView.h"

struct BumperMeasurements {
    double w{};
    double h{};
    cv::Rect rect;
    cv::RotatedRect rotated_rect;
};

struct Position3D {
    double z_cm;
};

// Global tracker instance
static RobotTracker g_tracker;

int hammingDistance(const std::string &str1, const std::string &str2) {
    if (str1.length() != str2.length()) {
        return 4;
    }

    int count = 0;
    for (size_t i = 0; i < str1.length(); ++i) {
        if (str1[i] != str2[i]) {
            count++;
        }
    }

    return count;
}

int levenshteinDist(const std::string &word1, const std::string &word2) {
    const int size1 = static_cast<int>(word1.size());
    const int size2 = static_cast<int>(word2.size());
    std::vector<std::vector<int> > verif(size1 + 1, std::vector<int>(size2 + 1));

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
            int cost = (word2[j - 1] == word1[i - 1]) ? 0 : 1;
            verif[i][j] = std::min(
                std::min(verif[i - 1][j] + 1, verif[i][j - 1] + 1),
                verif[i - 1][j - 1] + cost
            );
        }
    }

    return verif[size1][size2];
}

template<typename T>
T findMode(const std::vector<T> &data) {
    std::unordered_map<T, int> counts;
    for (const T &value: data) {
        ++counts[value];
    }

    int maxCount = 0;
    T mostFrequentVal = T();

    for (const auto &pair: counts) {
        if (pair.second > maxCount) {
            maxCount = pair.second;
            mostFrequentVal = pair.first;
        }
    }

    return mostFrequentVal;
}

BumperMeasurements getMeasurementsFromContour(const std::vector<cv::Point> &contour) {
    BumperMeasurements m;
    m.rect = boundingRect(contour);
    m.rotated_rect = cv::minAreaRect(contour);

    m.w = std::min(static_cast<int>(m.rotated_rect.size.width), m.rect.width);
    m.h = std::min(static_cast<int>(m.rotated_rect.size.height), m.rect.height);

    return m;
}

Position3D getPosition3D(
    const BumperMeasurements &m,
    const double focal_length_cm = 0.36,
    const double known_height_cm = 10.6,
    const double pixel_height_cm = 0.0003
) {
    Position3D pos{};
    pos.z_cm = (known_height_cm * focal_length_cm) / (m.h * pixel_height_cm);
    return pos;
}

void drawMeasurements(
    cv::Mat &frame,
    const BumperMeasurements &m,
    const Position3D &pos,
    const std::string &robot_label = "",
    const std::string &prev_label = "",
    const cv::Scalar &robot_color = cv::Scalar(0, 255, 0)
) {
    static std::unordered_map<std::string, kalmanFilter> filters;
    kalmanFilter &filter = filters[""];
    if (prev_label == robot_label) {
        filter = filters[robot_label];
    }
    if (robot_label.empty() && !prev_label.empty()) {
        filter = filters[prev_label];
    }

    constexpr double SCREEN_WIDTH = 1280;
    constexpr double SCREEN_HEIGHT = 720;
    constexpr double X_FOV = 70;
    constexpr double Y_FOV = 43;

    std::stringstream ss;

    cv::Point robot_center = cv::Point(static_cast<int>(m.rotated_rect.center.x - SCREEN_WIDTH / 2),
                                       static_cast<int>(m.rotated_rect.center.y - SCREEN_HEIGHT / 2));

    constexpr double max_cord_x = SCREEN_WIDTH / 2;
    constexpr double max_cord_y = SCREEN_HEIGHT / 2;

    const double x_angle = (X_FOV / 2 * (robot_center.x / max_cord_x)) * CV_PI / 180.0;
    const double y_angle = (Y_FOV / 2 * (robot_center.y / max_cord_y)) * CV_PI / 180.0;

    const double x_coordinate = pos.z_cm * cos(x_angle) / 100;
    const double y_coordinate = pos.z_cm * sin(x_angle) * cos(y_angle) / 100;
    const double z_coordinate = pos.z_cm * sin(x_angle) * sin(y_angle) / 100;
    cv::Vec3d filtered;
    filtered[0] = x_coordinate;
    filtered[1] = y_coordinate;
    filtered[2] = z_coordinate;
    if (robot_label.empty() && prev_label.empty()) {
        filtered = filter.update(x_coordinate, y_coordinate, z_coordinate, static_cast<double>(1) / 5);
    }

    // Update tracker with robot position
    if (!robot_label.empty()) {
        g_tracker.updateRobotPosition(filtered[0], filtered[1], filtered[2], robot_label, robot_color);
    }

    ss << std::fixed << std::setprecision(2)
            << "(" << filtered[0] << "m, " << filtered[1] << "m, " << filtered[2] << "m)";

    cv::putText(frame, ss.str(),
                cv::Point(m.rect.x + 10, m.rect.y - 10),
                cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0, 255, 255), 2);
}

void detectEdgesBumper(
    cv::Mat &blankFrame,
    const std::string teamNumbers[5],
    cv::Mat &frame,
    std::vector<Detection> &detections
) {
    std::vector<std::vector<cv::Point> > contours, overlappingContoursRed, overlappingContoursBlue;
    cv::Mat gray, edgesBlue, edgesRed, bMask, rMask, rMask1, hsv;

    cv::cvtColor(frame, hsv, cv::COLOR_BGR2HSV);

    cv::inRange(hsv, cv::Scalar(100, 120, 70), cv::Scalar(130, 255, 255), bMask);
    cv::inRange(hsv, cv::Scalar(0, 100, 100), cv::Scalar(10, 255, 255), rMask);
    cv::inRange(hsv, cv::Scalar(160, 100, 100), cv::Scalar(179, 255, 255), rMask1);

    rMask = rMask | rMask1;

    cv::Canny(rMask, edgesRed, 120, 255);
    cv::findContours(edgesRed, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

    for (const auto &contour: contours) {
        cv::Rect contourRect = cv::boundingRect(contour);

        for (const auto &det: detections) {
            if ((contourRect & det.bounding_box) == contourRect) {
                overlappingContoursRed.emplace_back(contour);
                break;
            }
        }
    }

    cv::Canny(bMask, edgesBlue, 120, 255);
    cv::findContours(edgesBlue, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
    std::string label;
    for (const auto &contour: contours) {
        cv::Rect contourRect = cv::boundingRect(contour);

        for (const auto &det: detections) {
            if ((contourRect & det.bounding_box) == contourRect) {
                overlappingContoursBlue.emplace_back(contour);
                break;
            }
        }
    }

    cv::Mat overlapMaskRed = cv::Mat::zeros(edgesRed.size(), CV_8UC1);
    cv::Mat overlapMaskBlue = cv::Mat::zeros(edgesBlue.size(), CV_8UC1);
    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(50, 25));

    cv::GaussianBlur(overlapMaskRed, overlapMaskRed, cv::Size(15, 5), 1);
    cv::GaussianBlur(overlapMaskBlue, overlapMaskBlue, cv::Size(15, 5), 1);

    cv::drawContours(overlapMaskRed, overlappingContoursRed, -1, cv::Scalar(255), cv::FILLED);
    cv::morphologyEx(overlapMaskRed, overlapMaskRed, cv::MORPH_CLOSE, kernel);
    cv::findContours(overlapMaskRed, overlappingContoursRed, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

    cv::drawContours(overlapMaskBlue, overlappingContoursBlue, -1, cv::Scalar(255), cv::FILLED);
    cv::morphologyEx(overlapMaskBlue, overlapMaskBlue, cv::MORPH_CLOSE, kernel);
    cv::findContours(overlapMaskBlue, overlappingContoursBlue, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

    // Clear previous robot positions
    g_tracker.clearRobots();

    int robot_id = 0;

    findNumbers(detections, blankFrame, frame, teamNumbers);

    // Process red robots
    for (const auto &contour: overlappingContoursRed) {
        if (contourArea(contour) < 250) continue;
        BumperMeasurements m = getMeasurementsFromContour(contour);
        Position3D pos = getPosition3D(m);
        std::string prevLabel = label;
        for (auto &det: detections) {
            if ((det.bounding_box & boundingRect(contour)) == boundingRect(contour)) {
                label = "Red-" + det.label;
            }
        }
        drawMeasurements(frame, m, pos, label, prevLabel, cv::Scalar(0, 0, 255));
    }

    // Process blue robots
    for (const auto &contour: overlappingContoursBlue) {
        if (contourArea(contour) < 250) continue;
        BumperMeasurements m = getMeasurementsFromContour(contour);
        Position3D pos = getPosition3D(m);
        std::string prevLabel = label;
        for (auto &det: detections) {
            if ((det.bounding_box & boundingRect(contour)) == boundingRect(contour)) {
                label = "Blue-" + det.label;
            }
        }
        drawMeasurements(frame, m, pos, label, prevLabel, cv::Scalar(255, 0, 0));
    }

    cv::drawContours(frame, overlappingContoursBlue, -1, cv::Scalar(255, 0, 0), 2);
    cv::drawContours(frame, overlappingContoursRed, -1, cv::Scalar(0, 0, 255), 2);

    // Render top-down view
    g_tracker.render();

    cv::imshow("detectEdgesBumper", frame);
}

void findNumbers(std::vector<Detection> &detections, const cv::Mat &blankFrame, cv::Mat &frame,
                 const std::string teamNumbers[5]) {
    auto *api = new tesseract::TessBaseAPI();
    if (api->Init("C:/Program Files/Tesseract-OCR/tessdata", "eng", tesseract::OEM_LSTM_ONLY)) {
        std::cerr << "Could not initialize tesseract." << std::endl;
        return;
    }

    for (auto &det: detections) {
        cv::Mat img = blankFrame(det.bounding_box).clone();

        cv::Mat hsv, colorMask;
        cv::cvtColor(img, hsv, cv::COLOR_BGR2HSV);
        cv::inRange(hsv, cv::Scalar(0, 0, 100), cv::Scalar(179, 50, 255), colorMask);

        cv::Mat gray;
        cv::cvtColor(img, gray, cv::COLOR_BGR2GRAY);

        if (gray.cols < 300) {
            double scale = 300.0 / gray.cols;
            cv::resize(gray, gray, cv::Size(), scale, scale, cv::INTER_CUBIC);
            cv::resize(colorMask, colorMask, cv::Size(), scale, scale, cv::INTER_CUBIC);
        }

        cv::Mat denoised;
        cv::bilateralFilter(gray, denoised, 9, 75, 75);

        cv::Mat binary;
        cv::adaptiveThreshold(denoised, binary, 255, cv::ADAPTIVE_THRESH_GAUSSIAN_C,
                              cv::THRESH_BINARY, 11, 2);

        cv::Mat final;
        cv::bitwise_and(binary, colorMask, final);

        cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(2, 2));
        cv::morphologyEx(final, final, cv::MORPH_CLOSE, kernel);

        api->SetPageSegMode(tesseract::PSM_SPARSE_TEXT_OSD);
        api->SetVariable("tessedit_char_whitelist", "0123456789");
        api->SetImage(final.data, final.cols, final.rows, 1, final.step);

        char *outText = api->GetUTF8Text();
        std::string result(outText);
        std::string sResult(outText);
        delete[] outText;

        result.erase(std::remove_if(result.begin(), result.end(), ::isspace), result.end());

        int minIndex = 0;

        if (!result.empty() && std::all_of(result.begin(), result.end(), ::isdigit)) {
            int minDist = INT_MAX;
            for (int i = 0; i < teamNumbers->size(); i++) {
                int d = levenshteinDist(sResult, teamNumbers[i]);
                if (d < minDist) {
                    minDist = d;
                    minIndex = i;
                }
            }
        }

        if (levenshteinDist(sResult, teamNumbers[minIndex]) >= 3) {
            api->End();
            delete api;
            return;
        }

        result = teamNumbers[minIndex];
        det.label = result;
        cv::putText(frame, result,
                    cv::Point(det.bounding_box.x + 10, det.bounding_box.y - 50),
                    cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0, 255, 0), 2);
    }

    api->End();
    delete api;
}
