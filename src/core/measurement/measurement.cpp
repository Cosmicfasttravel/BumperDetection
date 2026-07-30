#include "measurement.h"

#include <opencv2/core.hpp>
#include <opencv2/core/cvdef.h>
#include <opencv2/core/mat.hpp>
#include <opencv2/core/types.hpp>

#include "core/config/config_extraction.h"
#include "global/internal/built_components.h"


namespace measurements {
    enum class Color {
        RED,
        BLUE,
        NONE,
    };

    namespace {
        double getDistance(const double measured_height) {
            const Config config = config::getLatestCopy();

            const double focal_length = config.camera.focal_length; // Centimeters
            const double actual_height = config.bumper.height; // Centimeters
            const double pixel_height = config.camera.pixel_height; // Centimeters

            if (measured_height <= 0) return 0.0;
            return (actual_height * focal_length) / (measured_height * pixel_height);
        }
    }

    Position3D getXYZ(const cv::Rect bound, double measured_height) {
        const Config config = config::getLatestCopy();
        thread_local cv::Rect boundingBox = bound;

        const double SCREEN_HEIGHT = config.screen.height;
        const double SCREEN_WIDTH = config.screen.width;

        const double X_FOV = config.screen.x_fov;
        const double Y_FOV = config.screen.y_fov;

        const double max_cord_x = SCREEN_WIDTH / 2.0;
        const double max_cord_y = SCREEN_HEIGHT / 2.0;

        const double middle_bound_x = boundingBox.x + (0.5 * boundingBox.width);
        const double bottom_bound_y = boundingBox.y + boundingBox.height;

        const double offset_x = (middle_bound_x - max_cord_x) / max_cord_x;
        const double offset_y = (bottom_bound_y - max_cord_y) / max_cord_y;

        const double x_angle = (X_FOV / 2 * offset_x) * CV_PI / 180.0;
        const double y_angle = (Y_FOV / 2 * offset_y) * CV_PI / 180.0;

        const double distance = getDistance(measured_height);
        if (distance == 0) return {};

        Position3D position = {
            (distance / 100.0) * sin(y_angle) * cos(x_angle), // If these don't work try cos(y) * cos(x)
            (distance / 100.0) * sin(y_angle) * sin(x_angle), // cos(y) * sin(x)
            (distance / 100.0) * cos(x_angle) // sin(y)
        };

        return position;
    }

    double getHeight(const cv::Mat &hsv, const cv::Rect &box) {
        cv::Mat redMask, redMask1;
        cv::Mat blueMask;

        Color robotColor = Color::NONE;

        cv::Mat hsvFrame = hsv;
        cv::Rect boundingBox = box;

        auto bumperBoundingBox = hsvFrame(boundingBox);

        const Config config = config::getLatestCopy();

        //Red thresholds
        const auto lowerRedThreshold_1 = cv::Scalar(config.height_measurement.red_mask_thresholds_1.hue_lower,
                                                    config.height_measurement.red_mask_thresholds_1.saturation_lower,
                                                    config.height_measurement.red_mask_thresholds_1.value_lower);
        const auto upperRedThreshold_1 = cv::Scalar(config.height_measurement.red_mask_thresholds_1.hue_upper,
                                                    config.height_measurement.red_mask_thresholds_1.saturation_upper,
                                                    config.height_measurement.red_mask_thresholds_1.value_upper);

        const auto lowerRedThreshold_2 = cv::Scalar(config.height_measurement.red_mask_thresholds_2.hue_lower,
                                                    config.height_measurement.red_mask_thresholds_2.saturation_lower,
                                                    config.height_measurement.red_mask_thresholds_2.value_lower);
        const auto upperRedThreshold_2 = cv::Scalar(config.height_measurement.red_mask_thresholds_2.hue_upper,
                                                    config.height_measurement.red_mask_thresholds_2.saturation_upper,
                                                    config.height_measurement.red_mask_thresholds_2.value_upper);

        //Blue threshold
        const auto lowerBlueThreshold = cv::Scalar(config.height_measurement.blue_mask_thresholds.hue_lower,
                                                   config.height_measurement.blue_mask_thresholds.saturation_lower,
                                                   config.height_measurement.blue_mask_thresholds.value_lower);
        const auto upperBlueThreshold = cv::Scalar(config.height_measurement.blue_mask_thresholds.hue_upper,
                                                   config.height_measurement.blue_mask_thresholds.saturation_upper,
                                                   config.height_measurement.blue_mask_thresholds.value_upper);

        cv::inRange(bumperBoundingBox, lowerRedThreshold_1, upperRedThreshold_1, redMask);
        cv::inRange(bumperBoundingBox, lowerRedThreshold_2, upperRedThreshold_2, redMask1);
        cv::bitwise_or(redMask, redMask1, redMask);

        cv::inRange(bumperBoundingBox, lowerBlueThreshold, upperBlueThreshold, blueMask);

        auto centerX = boundingBox.x + boundingBox.width / 2;
        auto centerY = boundingBox.y + boundingBox.height / 2;

        int blueRelativeCenterX = centerX - boundingBox.x;
        int blueRelativeCenterY = centerY - boundingBox.y;

        int redRelativeCenterX = centerX - boundingBox.x;
        int redRelativeCenterY = centerY - boundingBox.y;

        redRelativeCenterX = std::clamp(redRelativeCenterX, 0, redMask1.cols - 1);
        redRelativeCenterY = std::clamp(redRelativeCenterY, 0, redMask1.rows - 1);

        blueRelativeCenterX = std::clamp(blueRelativeCenterX, 0, blueMask.cols - 1);
        blueRelativeCenterY = std::clamp(blueRelativeCenterY, 0, blueMask.rows - 1);

        // Make it sample multiple pixels instead of only 1
        if (redMask1.at<uchar>(redRelativeCenterY, redRelativeCenterX) > 0) robotColor = Color::RED;
        else if (blueMask.at<uchar>(blueRelativeCenterY, blueRelativeCenterX) > 0) robotColor = Color::BLUE;
        else robotColor = Color::NONE;

        if (robotColor == Color::NONE) return 0;

        double height = 0;

        auto topY = boundingBox.y;
        auto bottomY = boundingBox.y + boundingBox.height;


        //add random sampling for pixel height and ensure that the numbers are filled
        for (auto y = topY; y < bottomY; y++) {
            int relY = std::clamp(y - boundingBox.y, 0, (robotColor == Color::RED) ? redMask1.rows - 1 : blueMask.rows - 1);
            int c = 0;
            if (robotColor == Color::RED) c = redMask1.at<uchar>(relY, redRelativeCenterX);
            else if (robotColor == Color::BLUE) c = blueMask.at<uchar>(relY, blueRelativeCenterX);
            if (c > 0) height++;
        }
        return height;
    }

    int checkBounds(const cv::Mat &hsv, const cv::Rect &boundingBox) {
        //implement system to check if there are two robot next to eachother and make it callable without needing masks

        auto hsvRoi = hsv(boundingBox);

        cv::Mat redMask, redMask1;
        cv::Mat blueMask;

        const Config config = config::getLatestCopy();

        //Red thresholds
        const auto lowerRedThreshold_1 = cv::Scalar(config.height_measurement.red_mask_thresholds_1.hue_lower,
                                                    config.height_measurement.red_mask_thresholds_1.saturation_lower,
                                                    config.height_measurement.red_mask_thresholds_1.value_lower);
        const auto upperRedThreshold_1 = cv::Scalar(config.height_measurement.red_mask_thresholds_1.hue_upper,
                                                    config.height_measurement.red_mask_thresholds_1.saturation_upper,
                                                    config.height_measurement.red_mask_thresholds_1.value_upper);

        const auto lowerRedThreshold_2 = cv::Scalar(config.height_measurement.red_mask_thresholds_2.hue_lower,
                                                    config.height_measurement.red_mask_thresholds_2.saturation_lower,
                                                    config.height_measurement.red_mask_thresholds_2.value_lower);
        const auto upperRedThreshold_2 = cv::Scalar(config.height_measurement.red_mask_thresholds_2.hue_upper,
                                                    config.height_measurement.red_mask_thresholds_2.saturation_upper,
                                                    config.height_measurement.red_mask_thresholds_2.value_upper);

        //Blue threshold
        const auto lowerBlueThreshold = cv::Scalar(config.height_measurement.blue_mask_thresholds.hue_lower,
                                                   config.height_measurement.blue_mask_thresholds.saturation_lower,
                                                   config.height_measurement.blue_mask_thresholds.value_lower);
        const auto upperBlueThreshold = cv::Scalar(config.height_measurement.blue_mask_thresholds.hue_upper,
                                                   config.height_measurement.blue_mask_thresholds.saturation_upper,
                                                   config.height_measurement.blue_mask_thresholds.value_upper);
        cv::inRange(hsvRoi, lowerRedThreshold_1, upperRedThreshold_1, redMask);
        cv::inRange(hsvRoi, lowerRedThreshold_2, upperRedThreshold_2, redMask1);
        cv::bitwise_or(redMask, redMask1, redMask);

        cv::inRange(hsvRoi, lowerBlueThreshold, upperBlueThreshold, blueMask);

        int nearbyRobotCount = 0;

        for (int i = 0; i < 2; i++) {
            cv::Mat mask;
            switch (i) {
                case 0:
                    mask = redMask;
                    break;
                case 1:
                    mask = blueMask;
                    break;
                default: ;
            }

            if (countNonZero(mask) == 0)
                continue;

            double allowedError;

            double typicalAspectRatio = 3.15;
            if (boundingBox.height <= 0)
                continue;

            double measuredAspectRatio = static_cast<double>(boundingBox.width) / boundingBox.height;
            allowedError = typicalAspectRatio * 0.10;

            if (std::abs(measuredAspectRatio - typicalAspectRatio) > allowedError) {
                nearbyRobotCount++;
                continue;
            }

            double boundingBoxArea = boundingBox.width * boundingBox.height;
            double measuredArea = countNonZero(mask);
            double fillRatio = static_cast<double>(measuredArea) / boundingBoxArea;

            if (fillRatio <= 0.75) {
                nearbyRobotCount++;
                continue;
            }
        }

        return nearbyRobotCount;
    }
}
