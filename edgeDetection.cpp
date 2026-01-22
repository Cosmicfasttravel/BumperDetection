#include <iostream>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/core/mat.hpp>
#include <sstream>
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
};

struct Position3D {
    double z_cm;
};

namespace det {

    BumperMeasurements getMeasurementsFromContour(const std::vector<cv::Point>& contour) {
        BumperMeasurements m;
        m.rect = boundingRect(contour);

        m.w = m.rect.width;
        m.h = m.rect.height;

        return m;
    }

    Position3D getPosition3D(
        const BumperMeasurements& m,
        double focal_length,
        double known_height_cm
    ) {
        Position3D pos{};
        pos.z_cm = (known_height_cm * focal_length) / m.h;
        return pos;
    }

    void drawMeasurements(
        cv::Mat& frame,
        const BumperMeasurements& m,
        const Position3D& pos
    ) {

        // Display distance
        std::stringstream ss;
        ss << std::fixed << std::setprecision(2)
           << pos.z_cm / 100.0 << "m";

        cv::putText(frame, ss.str(),
            cv::Point(m.rect.x + 10, m.rect.y - 10),
            cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0, 255, 255), 2);
    }

    void detectEdgesBumper(
        cv::Mat& blankFrame,
        cv::Mat& frame,
        const std::vector<Detection>& detections,
        double focal_length = 400, //robotsecond vid = ~400, robotcropped = ~1400
        double bumper_height_cm = 11
    ) {
        std::vector<std::vector<cv::Point>> contours, overlappingContours;
        cv::Mat gray, edges, bMask, rMask, rMask1, hsv;

        cv::cvtColor(frame, hsv, cv::COLOR_BGR2HSV);

        cv::inRange(hsv, cv::Scalar(100, 120, 70), cv::Scalar(130, 255, 255), bMask);
        cv::inRange(hsv, cv::Scalar(0, 100, 100), cv::Scalar(10, 255, 255), rMask);
        cv::inRange(hsv, cv::Scalar(160, 100, 100), cv::Scalar(179, 255, 255), rMask1);

        rMask = rMask | rMask1;
        cv::bitwise_or(bMask, rMask, gray);

        cv::Canny(gray, edges, 100, 255);

        cv::findContours(edges, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

        for (const auto& contour : contours) {
            cv::Rect contourRect = cv::boundingRect(contour);

            for (const auto& det : detections) {
                if ((contourRect & det.bounding_box) == contourRect) {
                    overlappingContours.push_back(contour);
                    break;
                }
            }
        }

        cv::Mat overlapMask = cv::Mat::zeros(edges.size(), CV_8UC1);

        cv::drawContours(
            overlapMask,
            overlappingContours,
            -1,
            cv::Scalar(255),
            cv::FILLED
        );

        cv::Mat kernel = cv::getStructuringElement(
            cv::MORPH_ELLIPSE, cv::Size(50, 50));

        cv::morphologyEx(
            overlapMask,
            overlapMask,
            cv::MORPH_CLOSE,
            kernel
        );

        cv::findContours(
            overlapMask,
            overlappingContours,
            cv::RETR_EXTERNAL,
            cv::CHAIN_APPROX_SIMPLE
        );

        // Process each contour and calculate distance
        for (const auto& contour : overlappingContours) {
            if (contourArea(contour) < 600) continue;
            BumperMeasurements m = getMeasurementsFromContour(contour);

            Position3D pos = getPosition3D(m, focal_length, bumper_height_cm);
            drawMeasurements(frame, m, pos);
        }

        cv::drawContours(frame, overlappingContours, -1, cv::Scalar(255, 0, 255), 2);

        cv::imshow("detectEdgesBumper", frame);
    }
}

/*
CALIBRATION INSTRUCTIONS:
-------------------------
Your focal_length = 800 is likely WRONG. Here's how to calibrate:

1. Measure your bumper's actual width with a ruler
   Example: 101.6 cm (40 inches)

2. Place the robot at a KNOWN distance from camera
   Example: 300 cm (10 feet) away

3. Run this code and look at the console output for "Width: XXX px"
   Example: you see "Width: 270 px"

4. Calculate your correct focal length:
   focal_length = (pixel_width × distance) / real_width
   focal_length = (270 × 300) / 101.6 = 797.2

5. Update the default parameter in detectEdgesBumper():
   double focal_length = 797.2  // Your calibrated value

TIPS:
- Calibrate at a medium distance (200-400 cm works well)
- Make sure the bumper is straight-on to the camera
- The blue line should now consistently track the longest dimension
- Distance should be accurate regardless of robot rotation
*/