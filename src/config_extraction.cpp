#include "config_extraction.h"

#include <filesystem>
#include <functional>
#include <iostream>

static const std::filesystem::path CONFIG_PATH = "../config.txt";
std::vector<std::string> fileContents = {};
std::filesystem::file_time_type prevTime = {};

static Config r_config;

void extractAll()
{
    fileContents.clear();
    std::ifstream config(CONFIG_PATH);
    if (!config.is_open()) {
        std::cerr << "Could not open " << CONFIG_PATH << std::endl;
        throw std::runtime_error("Could not open config file");
    }

    std::string s;
    while (std::getline(config, s))
    {
        fileContents.emplace_back(s);
    }

    r_config.bumper_height = std::stod(extractByTag("<bumper_height>"));
    r_config.focal_length = std::stod(extractByTag("<focal_length>"));
    r_config.pixel_height = std::stod(extractByTag("<pixel_height>"));

    r_config.screen_width = std::stod(extractByTag("<screen_width>"));
    r_config.screen_height = std::stod(extractByTag("<screen_height>"));
    r_config.x_fov = std::stod(extractByTag("<x_fov>"));
    r_config.y_fov = std::stod(extractByTag("<y_fov>"));

    r_config.avg_fps = std::stod(extractByTag("<avg_fps>"));
    r_config.process_noise = std::stod(extractByTag("<process_noise>"));
    r_config.measurement_noise = std::stod(extractByTag("<measurement_noise>"));
    r_config.error = std::stod(extractByTag("<error>"));

    r_config.lev_distance = std::stod(extractByTag("<lev_distance>"));

    r_config.ip = extractByTag("<ip>");

    r_config.conf_threshold = std::stod(extractByTag("<conf_threshold>"));
    r_config.nms_threshold = std::stod(extractByTag("<nms_threshold>"));

    r_config.rotation = std::stoi(extractByTag("<rotation>"));

    r_config.teams[0] = extractByTag("<t1>");
    r_config.teams[1] = extractByTag("<t2>");
    r_config.teams[2] = extractByTag("<t3>");
    r_config.teams[3] = extractByTag("<t4>");
    r_config.teams[4] = extractByTag("<t5>");

    r_config.loggingMode = std::stoi(extractByTag("<logging_mode>"));

    r_config.brightness = std::stoi(extractByTag("<brightness>"));
    r_config.contrast = std::stoi(extractByTag("<contrast>"));
    r_config.hue = std::stoi(extractByTag("<hue>"));
    r_config.saturation = std::stoi(extractByTag("<saturation>"));
    r_config.gain = stoi(extractByTag("<gain>"));
    r_config.exposure = std::stoi(extractByTag("<exposure>"));

}

bool pollForChanges()
{
    if (auto curTime = std::filesystem::last_write_time(CONFIG_PATH); prevTime != curTime)
    {
        prevTime = curTime;
        extractAll();

        return true;
    }
    return false;
}

std::string extractByTag(const std::string &tag)
{
    for (const auto &line : fileContents)
    {
        if (auto pos = line.find(tag); pos != std::string::npos)
        {
            std::cout << "Found tag '" << tag << "'.At line " << pos << std::endl;
            return line.substr(pos + tag.length());
        }
    }
    std::cerr << "WARNING: Could not find tag " << tag << std::endl;
    return "";
}

const Config &getConfig()
{
    return r_config;
}