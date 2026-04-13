#include "config_extraction.h"

#include <filesystem>
#include <functional>
#include <ostream>
#include <fstream>
#include <iostream>
#include "debug_log.h"
#include <nlohmann/json.hpp>
using json = nlohmann::json;

Config r_config;

static const std::filesystem::path CONFIG_PATH = "../config.json";
std::ifstream file(CONFIG_PATH);
json data = json::parse(file);

std::filesystem::file_time_type prevTime = {};

void extract(){
    r_config = data.get<Config>();

    r_config.teams.blueTeams = data["blueTeams"];
    r_config.teams.redTeams = data["redTeams"];

    r_config.yolo.output_dimensions = data["yolo"]["output_dimensions"];
}

bool pollForChanges()
{
    auto curTime = std::filesystem::last_write_time(CONFIG_PATH);

    if (prevTime != curTime)
    {
        prevTime = curTime;

        try
        {
            std::ifstream file(CONFIG_PATH);
            if (!file.is_open())
            {
                std::cerr << "Failed to open config\n";
                logger->warn("Failed to open config");
                return false;
            }

            data = json::parse(file);

            std::cout << "File reread\n";
            logger->info("Reread file");
            extract();

            return true;
        }
        catch (const std::exception &e)
        {
            std::cerr << "JSON failed: " << e.what() << "\n";
            logger->info("JSON failed");
            return false;
        }
    }

    return false;
}

Config& getConfig(){
    return r_config;
}



