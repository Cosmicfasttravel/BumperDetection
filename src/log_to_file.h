#ifndef BUMPERDETECTION_LOG_TO_FILE_H
#define BUMPERDETECTION_LOG_TO_FILE_H
#include <string>
#include <fstream>
#include <ios>
#include "config_extraction.h"

namespace Logger
{
    extern Config config;
    extern std::ofstream logFile;
}

void initLogFile(const std::string &fileName, const Config &r_config);

template <typename T>
void logToFile(const std::string& tag, const T& value) {
    if (Logger::config.loggingMode && !Logger::config.displayMode) Logger::logFile << tag << ": " << value << "\n";
}

void logFPS(std::string fps);

void closeLogFile();

#endif // BUMPERDETECTION_LOG_TO_FILE_H
