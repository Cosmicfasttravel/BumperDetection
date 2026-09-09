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
    int frame_skip;
};

struct Kalman
{
    double process_noise;
    double measurement_noise;
    double error;
};

struct MaskThresholds
{
    int hue_lower;
    int saturation_lower;
    int value_lower;

    int hue_upper;
    int saturation_upper;
    int value_upper;
};

struct OCR
{
    double lev_distance;
    int max_instances;
    int morphology_kernel_size;
    std::string mode;
    int min_img_size;

    MaskThresholds mask_thresholds;

    std::string tessdata_path;
};

struct HeightMeasurement {
    MaskThresholds red_mask_thresholds_1;
    MaskThresholds red_mask_thresholds_2;
    MaskThresholds blue_mask_thresholds;
};

struct NetworkTables
{
    std::string ip;
};

struct Yolo
{
    double conf_threshold;
    double nms_threshold;
    int input_dimensions;
    std::array<int, 3> output_dimensions;
};

struct Teams
{
    std::array<std::string, 3> blueTeams;
    std::array<std::string, 3> redTeams;
};

struct Modes
{
    bool write_frame_to_file;
    bool video;
    bool display;
    bool dynamic_camera_properties_updating;
    bool cam_tuning;
    mutable bool logging;
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
    int id_count;
    int lost_threshold;
};

struct InputPaths {
    std::string rknn_path;
    std::string onnx_path;
    std::string video_path;
};

struct Config
{
    Bumper bumper;
    Screen screen;
    Kalman position_kalman;
    OCR ocr;
    HeightMeasurement height_measurement;
    int thread_pool_size;
    NetworkTables nt;
    Yolo yolo;
    Teams teams;
    Modes modes;
    Camera camera;
    Tracking tracking;
    InputPaths input_paths;
};

NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(Bumper,
    height
)

NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(Screen,
    width,
    height,
    x_fov,
    y_fov,
    rotation,
    frame_skip
)

NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(Kalman,
    process_noise,
    measurement_noise,
    error
)

NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(MaskThresholds,
    hue_lower,
    hue_upper,
    saturation_lower,
    saturation_upper,
    value_lower,
    value_upper
)

NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(OCR,
    lev_distance,
    max_instances,
    morphology_kernel_size,
    mode,
    min_img_size,
    mask_thresholds,
    tessdata_path
)

NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(HeightMeasurement,
    red_mask_thresholds_1,
    red_mask_thresholds_2,
    blue_mask_thresholds
)

NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(NetworkTables,
    ip
)

NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(Yolo,
    conf_threshold,
    nms_threshold,
    input_dimensions
)

NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(Modes,
    write_frame_to_file,
    video,
    display,
    dynamic_camera_properties_updating,
    cam_tuning,
    logging
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
    max_distance_threshold_y,
    lost_threshold,
    id_count
)

NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(InputPaths,
    rknn_path,
    onnx_path,
    video_path
)

NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(Config,
    bumper,
    screen,
    position_kalman,
    ocr,
    height_measurement,
    thread_pool_size,
    nt,
    yolo,
    modes,
    camera,
    tracking,
    input_paths
)

Config &getConfig();

bool pollForChanges();
void extractConfig();

#endif // BUMPERDETECTION_CONFIG_EXTRACTION_H