#include "log_to_file.h"
#include <iostream>

namespace Logger
{
    Config config;
    std::ofstream logFile;
}

void initLogFile(const std::string &fileName, const Config &r_config)
{
    Logger::config = r_config;
    Logger::logFile = std::ofstream("../" + fileName + ".txt", std::ios_base::app);
}

void logFPS(std::string fps){
    static size_t counter = 1;
    if(counter >= 10000) std::exit(0);

    Logger::logFile << counter << " " << fps << std::endl;
    std::cout << counter << " " << fps << std::endl;
    counter++;
}

void closeLogFile()
{
    Logger::logFile.close();
}