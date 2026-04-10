#ifndef BUMPERDETECTION_DEBUG_LOG_H
#define BUMPERDETECTION_DEBUG_LOG_H
#include <fstream>
#include <filesystem>
#include "config_extraction.h"

static const std::filesystem::path DEBUG_PATH = "../debug_log.txt";
inline std::ofstream dFile(DEBUG_PATH);

enum WarningType
{
    ERROR,
    WARNING,
    INFO,
    VERBOSE
};

template <typename t>
void logWarning(t warn, WarningType type = WARNING)
{
    if(!getConfig().modes.logging) return;

    dFile << "[";
    if(type == WARNING) dFile << "WARNING";
    if(type == INFO) dFile << "INFO";
    if(type == ERROR) dFile << "ERROR";
    if(type == VERBOSE) dFile << "VERBOSE";
    dFile << "]: " << warn << std::endl;
}

#endif