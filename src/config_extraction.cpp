#include "config_extraction.h"

std::string extractByTag(std::string tag) {
    std::ifstream config("../config.txt");

    std::string line;
    std::string sub;
    while (std::getline(config, line)){
        int pos = line.find(tag);
        if (pos != std::string::npos) {
            sub = line.substr(pos + tag.length(), line.length() - tag.length());
        }
    }
    return sub;
}