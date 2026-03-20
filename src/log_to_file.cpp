#include "log_to_file.h"
#include <fstream>
#include <ios>

std::ofstream logFile;
Config config;

void initLogFile(const std::string &fileName, Config &r_config) {
    config = r_config;
    logFile = std::ofstream("../" + fileName + ".txt", std::ios_base::app);
}

template <typename type>
void logToFile(const std::string& tag, const type& value) {
    if (config.loggingMode) logFile << tag << ": " << value << "\n";
}

void closeLogFile() {
    logFile.close();
}