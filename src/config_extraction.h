#ifndef BUMPERDETECTION_CONFIG_EXTRACTION_H
#define BUMPERDETECTION_CONFIG_EXTRACTION_H
#include <filesystem>
#include <string>
#include <array>
#include <nlohmann/json.hpp>

struct Bumper
{
    double height;
};

struct Screen
{
    double width;
    double height;
    double x_fov;
    double y_fov;

    int rotation;
};

struct Kalman
{
    double avg_fps;
    double process_noise;
    double measurement_noise;
    double error;
};

struct OCR
{
    double lev_distance;
    int max_instances;
    int morphology_kernel_size;
    std::string mode;
    int min_img_size;
};

struct NetworkTables
{
    std::string ip;
};

struct Yolo
{
    double conf_threshold;
    double nms_threshold;
};

struct Teams
{
    std::array<std::string, 3> blueTeams;
    std::array<std::string, 3> redTeams;
};

struct Modes
{
    bool logging;
    bool write_frame;
    bool video;
    bool display;
};

struct Camera
{
    int brightness;
    int exposure;
    int gain;
    int hue;
    int saturation;
    int contrast;
    int temperature;
    int auto_white_balance;
    int initial_blur;
    double focal_length;
    double pixel_height;
};

struct Tracking
{
    int max_distance_threshold_x;
    int max_distance_threshold_y;
};

struct Config
{
    Bumper bumper;
    Screen screen;
    Kalman kalman;
    OCR ocr;
    NetworkTables nt;
    Yolo yolo;
    Teams teams;
    Modes modes;
    Camera camera;
    Tracking tracking;
};

NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(Bumper,
    height
)

NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(Screen,
    width,
    height,
    x_fov,
    y_fov,
    rotation
)

NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(Kalman,
    avg_fps,
    process_noise,
    measurement_noise,
    error
)

NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(OCR,
    lev_distance,
    max_instances,
    morphology_kernel_sizes,
    teseract_mode,
    min_img_size
)

NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(NetworkTables,
    ip
)

NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(Yolo,
    conf_threshold,
    nms_threshold
)

NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(Modes,
    logging,
    write_frame,
    video,
    display
)

NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(Camera,
    brightness,
    exposure,
    gain,
    hue,
    saturation,
    contrast,
    temperature,
    auto_white_balance,
    initial_blur,
    focal_length,
    pixel_height
)

NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(Tracking,
    max_distance_threshold_x,
    max_distance_threshold_y
)

NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(Config,
    bumper,
    screen,
    kalman,
    ocr,
    nt,
    yolo,
    modes,
    camera,
    tracking
)

Config &getConfig();

bool pollForChanges();
void extract();

#endif // BUMPERDETECTION_CONFIG_EXTRACTION_H