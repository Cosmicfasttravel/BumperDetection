#include "measurement.h"

#include <opencv2/core.hpp>
#include <opencv2/core/cvdef.h>
#include <opencv2/core/mat.hpp>
#include <opencv2/core/types.hpp>
#include "core/detection/structure/detection_data.h"

#include "core/config/config_extraction.h"
#include "global/enums.h"
#include "global/internal/built_components.h"


namespace measurements {
    Position3D getXYZ(const Detection &detection, const double measured_height) {
        static Config config = config::getLatestCopy();
        if (config::checkConfigVersion(config)) config = config::getLatestCopy();

        thread_local cv::Rect boundingBox = detection.boundingBox;

        double max_cord_x = 1280.0 / 2.0;
        double max_cord_y = 720.0 / 2.0;

        double abs_bounding_x = boundingBox.x + (0.5 * boundingBox.width);
        double abs_bounding_y = boundingBox.y + (boundingBox.height);

        double fx = max_cord_x / tan((config.screen.x_fov / 2.0) * CV_PI / 180.0);
        double fy = max_cord_y / tan((config.screen.y_fov / 2.0) * CV_PI / 180.0);

        double pixel_offset_x = abs_bounding_x - max_cord_x;
        double pixel_offset_y = abs_bounding_y - max_cord_y;

        double d_left_right = pixel_offset_x / fx;
        double d_up_down = pixel_offset_y / fy;
        double d_forward = 1.0;

        double ray_length = sqrt(d_forward * d_forward + d_left_right * d_left_right + d_up_down * d_up_down);

        double total_dist_m = measured_height / 100.0;

        Position3D position = {
            total_dist_m * (d_forward / ray_length), // depth
            total_dist_m * (d_left_right / ray_length), // left-right
            total_dist_m * (d_up_down / ray_length) // up-down
        };

        return position;
    }

    double getHeight(const cv::Mat &hsv, const Detection &detection) {
        cv::Mat redMask, redMask1;
        cv::Mat blueMask;

        const cv::Mat& hsvFrame = hsv;
        cv::Rect boundingBox = detection.boundingBox;

        auto bumperBoundingBox = hsvFrame(boundingBox);

        static Config config = config::getLatestCopy();
        if (config::checkConfigVersion(config)) config = config::getLatestCopy();

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

        int blueRelativeCenterX = centerX - boundingBox.x;

        int redRelativeCenterX = centerX - boundingBox.x;

        redRelativeCenterX = std::clamp(redRelativeCenterX, 0, redMask1.cols - 1);

        blueRelativeCenterX = std::clamp(blueRelativeCenterX, 0, blueMask.cols - 1);

        double height = 0;

        auto topY = boundingBox.y;
        auto bottomY = boundingBox.y + boundingBox.height;


        //add random sampling for pixel height and ensure that the numbers are filled
        for (auto y = topY; y < bottomY; y++) {
            int relY = std::clamp(y - boundingBox.y, 0,
                                  (detection.color == Color::RED) ? redMask1.rows - 1 : blueMask.rows - 1);
            int c = 0;
            if (detection.color == Color::RED) c = redMask1.at<uchar>(relY, redRelativeCenterX);
            else if (detection.color == Color::BLUE) c = blueMask.at<uchar>(relY, blueRelativeCenterX);
            if (c > 0) height++;
        }
        return height;
    }

    Color calculateColor(const cv::Mat &hsv, const Detection &detection) {
        cv::Mat redMask, redMask1;
        cv::Mat blueMask;

        const cv::Rect& boundingBox = detection.boundingBox;

        auto bumperBoundingBox = hsv(boundingBox);

        static Config config = config::getLatestCopy();
        if (config::checkConfigVersion(config)) config = config::getLatestCopy();

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

        double redRatio =
                countNonZero(redMask) /
                static_cast<double>(boundingBox.area());

        double blueRatio =
                countNonZero(blueMask) /
                static_cast<double>(boundingBox.area());

        if (redRatio >= blueRatio && redRatio >= 0.1) return Color::RED; // needs config for red and blue ratio
        if (redRatio <= blueRatio && blueRatio >= 0.1) return Color::BLUE;
        return Color::NONE;

    }


    //Don't use this
    int checkBounds(const cv::Mat &hsv, const cv::Rect &boundingBox) {
        //implement system to check if there are two robot next to eachother and make it callable without needing masks

        auto hsvRoi = hsv(boundingBox);

        cv::Mat redMask, redMask1;
        cv::Mat blueMask;

        static Config config = config::getLatestCopy();
        if (config::checkConfigVersion(config)) config = config::getLatestCopy();

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

    cv::Rect trimBoundingBox(const cv::Mat& hsv, const Detection& detection) {
        auto hsvRoi = hsv(detection.boundingBox);

        cv::Mat redMask, redMask1;
        cv::Mat blueMask;

        static Config config = config::getLatestCopy();
        if (config::checkConfigVersion(config)) config = config::getLatestCopy();

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

        std::vector<cv::Point> points;

        if (detection.color == Color::RED) {
            cv::findNonZero(redMask, points);
        }
        else if (detection.color == Color::BLUE) {
            cv::findNonZero(blueMask, points);
        }
        else {
            return {};
        }

        cv::Rect trimmed = cv::boundingRect(points);

        trimmed.x += detection.boundingBox.x;
        trimmed.y += detection.boundingBox.y;

        double aspect = static_cast<double>(trimmed.width) / trimmed.height;
        if (aspect < 2.0 || aspect > 6.0) {
            return {};
        }

        return trimmed;
    }
}
