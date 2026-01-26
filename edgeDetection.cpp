#include <iostream>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/core/mat.hpp>
#include <sstream>
#include <cmath>
#include <iomanip>

struct Detection {
    int class_id;
    float confidence;
    cv::Rect bounding_box;
    std::string class_name;
};

struct BumperMeasurements {
    double w{};
    double h{};
    cv::Rect rect;
    cv::RotatedRect rotated_rect;
};

struct Position3D {
    double z_cm;
};

namespace det {

    BumperMeasurements getMeasurementsFromContour(const std::vector<cv::Point>& contour) {
        BumperMeasurements m;
        m.rect = boundingRect(contour);
        m.rotated_rect = cv::minAreaRect(contour);

        m.w = m.rotated_rect.size.width;
        m.h = m.rotated_rect.size.height;

        return m;
    }

    Position3D getPosition3D(
        const BumperMeasurements& m,
        double focal_length_cm = 0.36,
        double known_height_cm = 10.6,
        double pixel_height_cm = 0.0003
    ) {
        Position3D pos{};
        pos.z_cm = (known_height_cm * focal_length_cm) / (m.h * pixel_height_cm);
        return pos;
    }

    void drawMeasurements(
        cv::Mat& frame,
        const BumperMeasurements& m,
        const Position3D& pos
    ) {
        constexpr double SCREEN_WIDTH = 1280;
        constexpr double SCREEN_HEIGHT = 720;
        constexpr double X_FOV = 70;
        constexpr double Y_FOV = 43;


        std::stringstream ss;
        // cv::rectangle(frame, m.rect.tl(), m.rect.br(), cv::Scalar(255, 255, 255), 3); //Draws rectangle surrounding the contour, non-rotated

        cv::Point robot_center = cv::Point(static_cast<int>(m.rotated_rect.center.x - SCREEN_WIDTH / 2), static_cast<int>(m.rotated_rect.center.y - SCREEN_HEIGHT / 2));

        double max_cord_x = SCREEN_WIDTH / 2;
        double max_cord_y = SCREEN_HEIGHT / 2;

        double x_angle = (X_FOV/2 * (robot_center.x / max_cord_x)) * CV_PI / 180.0;
        double y_angle = (Y_FOV/2 * (robot_center.y / max_cord_y)) * CV_PI / 180.0;

        double x_coordinate = pos.z_cm * cos(x_angle) / 100;
        double y_coordinate = pos.z_cm * sin(x_angle) * cos(y_angle) / 100;
        double z_coordinate = pos.z_cm * sin(x_angle) * sin(y_angle) / 100;


        ss << std::fixed << std::setprecision(2)
           << pos.z_cm / 100.0 << "m" << ", (" << x_coordinate << "m, " << y_coordinate << "m, " << z_coordinate << "m)";

        cv::putText(frame, ss.str(),
            cv::Point(m.rect.x + 10, m.rect.y - 10),
            cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0, 255, 255), 2);
    }

    void detectEdgesBumper(
        cv::Mat& blankFrame,
        cv::Mat& frame,
        const std::vector<Detection>& detections
    ) {
        std::vector<std::vector<cv::Point>> contours, overlappingContoursRed, overlappingContoursBlue;
        cv::Mat gray, edgesBlue, edgesRed, bMask, rMask, rMask1, hsv;

        cv::cvtColor(frame, hsv, cv::COLOR_BGR2HSV);

        cv::inRange(hsv, cv::Scalar(100, 120, 70), cv::Scalar(130, 255, 255), bMask);
        cv::inRange(hsv, cv::Scalar(0, 100, 100), cv::Scalar(10, 255, 255), rMask);
        cv::inRange(hsv, cv::Scalar(160, 100, 100), cv::Scalar(179, 255, 255), rMask1);

        rMask = rMask | rMask1;


        cv::Canny(rMask, edgesRed, 120, 255);
        cv::findContours(edgesRed, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

        for (const auto& contour : contours) {
            cv::Rect contourRect = cv::boundingRect(contour);

            for (const auto& det : detections) {
                if ((contourRect & det.bounding_box) == contourRect) {
                    overlappingContoursRed.push_back(contour);
                    break;
                }
            }
        }

        cv::Canny(bMask, edgesBlue, 120, 255);
        cv::findContours(edgesBlue, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

        for (const auto& contour : contours) {
            cv::Rect contourRect = cv::boundingRect(contour);

            for (const auto& det : detections) {
                if ((contourRect & det.bounding_box) == contourRect) {
                    overlappingContoursBlue.push_back(contour);
                    break;
                }
            }
        }

        cv::Mat overlapMaskRed = cv::Mat::zeros(edgesRed.size(), CV_8UC1);
        cv::Mat overlapMaskBlue = cv::Mat::zeros(edgesBlue.size(), CV_8UC1);
        cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(50, 25));

        cv::drawContours(overlapMaskRed, overlappingContoursRed, -1, cv::Scalar(255), cv::FILLED);
        cv::morphologyEx(overlapMaskRed, overlapMaskRed, cv::MORPH_CLOSE, kernel);
        cv::findContours(overlapMaskRed, overlappingContoursRed, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

        cv::drawContours(overlapMaskBlue, overlappingContoursBlue, -1, cv::Scalar(255), cv::FILLED);
        cv::morphologyEx(overlapMaskBlue, overlapMaskBlue, cv::MORPH_CLOSE, kernel);
        cv::findContours(overlapMaskBlue, overlappingContoursBlue, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

        for (const auto& contour : overlappingContoursRed) {
            if (contourArea(contour) < 100) continue;
            BumperMeasurements m = getMeasurementsFromContour(contour);
            Position3D pos = getPosition3D(m);
            drawMeasurements(frame, m, pos);
        }

        for (const auto& contour : overlappingContoursBlue) {
            if (contourArea(contour) < 100) continue;
            BumperMeasurements m = getMeasurementsFromContour(contour);
            Position3D pos = getPosition3D(m);
            drawMeasurements(frame, m, pos);
        }

        cv::drawContours(frame, overlappingContoursBlue, -1, cv::Scalar(255, 0, 0), 2);
        cv::drawContours(frame, overlappingContoursRed, -1, cv::Scalar(0, 0, 255), 2);

        cv::imshow("detectEdgesBumper", frame);
    }
}
