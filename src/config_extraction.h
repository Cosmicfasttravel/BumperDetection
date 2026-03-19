#ifndef BUMPERDETECTION_CONFIG_EXTRACTION_H
#define BUMPERDETECTION_CONFIG_EXTRACTION_H
#include <filesystem>
#include <string>

struct Config {
    // Bumper
    double bumper_height;
    double focal_length;
    double pixel_height;

    // Screen
    double screen_width;
    double screen_height;
    double x_fov;
    double y_fov;

    // Kalman Filter
    double avg_fps;
    double process_noise;
    double measurement_noise;
    double error;

    // Levenshtein
    double lev_distance;

    // Network Tables
    std::string ip;

    // Yolo
    double conf_threshold;
    double nms_threshold;

    // Frame
    int rotation;

    // Teams
    std::string teams[5];

    // Modes
    int loggingMode;
    int writeFrameMode;

    // Camera
    int brightness;
    int exposure;
    int gain;
    int hue;
    int saturation;
    int contrast;
    int temperature;
    int autoWhiteBalance;
};

std::string extractByTag(const std::string& tag);
bool pollForChanges();
void extractAll();
const Config& getConfig();

#endif //BUMPERDETECTION_CONFIG_EXTRACTION_H