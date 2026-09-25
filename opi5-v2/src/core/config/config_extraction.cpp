#include "core/config/config_extraction.h"

#include <filesystem>
#include <fstream>
#include <functional>
#include <iostream>

#include <nlohmann/json.hpp>

#include "log/logger.h"

using json = nlohmann::json;

namespace {
    const std::filesystem::path RELATIVE_CONFIG_PATH = "../opi5-v2/config.json";
    std::filesystem::path ABSOLUTE_CONFIG_PATH;

    Config rConfig;
    std::filesystem::file_time_type prevTime{};

    std::mutex lockMutex;
    int configVersion = 0;
}

std::once_flag absolutePathResolvedFlag;
void resolveAbsolutePath() {
    std::call_once(absolutePathResolvedFlag, [] {
        ABSOLUTE_CONFIG_PATH = std::filesystem::absolute(RELATIVE_CONFIG_PATH);
    });
}

bool load() {
    try {
        std::ifstream file(ABSOLUTE_CONFIG_PATH);

        if (!file.is_open()) {
            std::cerr << "Failed to open config\n";
            logging::write("Failed to open config", spdlog::level::warn);
            return false;
        }

        json data = json::parse(file);

        rConfig = data.get<Config>();

        rConfig.teams.blueTeams = data["blueTeams"];
        rConfig.teams.redTeams = data["redTeams"];

        prevTime = std::filesystem::last_write_time(ABSOLUTE_CONFIG_PATH);

        logging::write("Loaded config");

        configVersion++;

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

uint64_t getVersion() {
    std::lock_guard<std::mutex> lock(lockMutex);
    return configVersion;
}

namespace config {
    Config getLatestCopy() {
        std::lock_guard<std::mutex> lock(lockMutex);
        rConfig.version = configVersion;
        return rConfig;
    }

    bool checkConfigVersion(const Config& config) {
        return config.version != getVersion();
    }

    bool tryUpdate() {
        std::lock_guard<std::mutex> lock(lockMutex);
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
}

