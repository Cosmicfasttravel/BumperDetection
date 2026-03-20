#ifndef BUMPERDETECTION_LOG_TO_FILE_H
#define BUMPERDETECTION_LOG_TO_FILE_H
#include <string>

#include "config_extraction.h"

void initLogFile(const std::string& fileName, Config& r_config);

template <typename type>
void logToFile(const std::string& tag, const type& value);

void closeLogFile();

#endif //BUMPERDETECTION_LOG_TO_FILE_H
