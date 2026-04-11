#ifndef BUMPERDETECTION_DEBUG_LOG_H
#define BUMPERDETECTION_DEBUG_LOG_H
#include "spdlog/spdlog.h"
#include "spdlog/sinks/basic_file_sink.h"
#include <iostream>

inline std::shared_ptr<spdlog::logger> logger;

void initLogger();
#endif