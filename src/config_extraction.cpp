#include "config_extraction.h"

#include <filesystem>
#include <functional>
#include <ostream>
#include <fstream>
#include <iostream>
#include "logger.h"
#include <nlohmann/json.hpp>
using json = nlohmann::json;

namespace {
    const std::filesystem::path RELATIVE_CONFIG_PATH = "../config.json";
    std::filesystem::path ABSOLUTE_CONFIG_PATH = "";

    Config robotConfig;
    std::filesystem::file_time_type prevTime{};
}

std::once_flag absolutePathResolvedFlag;
void resolveAbsolutePath() {
    std::call_once(absolutePathResolvedFlag, [] {
        ABSOLUTE_CONFIG_PATH = std::filesystem::absolute(RELATIVE_CONFIG_PATH);
    });
}

static bool load() {
    try {
        std::ifstream file(ABSOLUTE_CONFIG_PATH);

        if (!file.is_open()) {
            std::cerr << "Failed to open config\n";
            logging::write("Failed to open config", spdlog::level::warn);
            return false;
        }

        json data = json::parse(file);

        robotConfig = data.get<Config>();

        robotConfig.teams.blueTeams = data["blueTeams"];
        robotConfig.teams.redTeams = data["redTeams"];
        robotConfig.yolo.output_dimensions = data["yolo"]["output_dimensions"];

        prevTime = std::filesystem::last_write_time(ABSOLUTE_CONFIG_PATH);

        logging::write("Loaded config");

        robotConfig.version++;

        return true;
    } catch (const std::exception &e) {
        std::cerr << "Config load failed: " << e.what() << '\n';
        logging::write("Config load failed", spdlog::level::err);

    } catch (...) {
        std::cerr << "Config load failed for an unknown reason\n";
        logging::write("Config load failed for an unknown reason", spdlog::level::critical);
    }
    return false;
}

bool tryUpdate() {
    resolveAbsolutePath();

    try {
        auto curTime = std::filesystem::last_write_time(ABSOLUTE_CONFIG_PATH);
        if (curTime != prevTime) {
            return load();
        }
    } catch (const std::exception &e) {
        std::cerr << "File watch failed: " << e.what() << '\n';
    }

    return false;
}

const Config &getRef() {
    return robotConfig;
}

uint64_t getVersion() {
    return robotConfig.version;
}
