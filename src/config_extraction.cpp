#include "config_extraction.h"

#include <filesystem>

static const std::filesystem::path CONFIG_PATH = "../config.txt";
std::vector<std::string> fileContents = {};
std::filesystem::file_time_type prevTime = {};

void extractAll() {
    fileContents.clear();
    std::ifstream config(CONFIG_PATH);
    std::string s;
    while (std::getline(config, s)) {
        fileContents.emplace_back(s);
    }
}

void pollForChanges() {
    if (auto curTime = std::filesystem::last_write_time(CONFIG_PATH); prevTime != curTime) {
        prevTime = curTime;
        extractAll();
    }
}

std::string extractByTag(const std::string& tag) {
    for (const auto& line : fileContents) {
        if (auto pos = line.find(tag); pos != std::string::npos) {
            return line.substr(pos + tag.length());
        }
    }
    return "";
}
