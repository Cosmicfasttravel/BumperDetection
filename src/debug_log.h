#ifndef BUMPERDETECTION_DEBUG_LOG_H
#define BUMPERDETECTION_DEBUG_LOG_H
#include "spdlog/spdlog.h"
#include "spdlog/sinks/basic_file_sink.h"
#include <iostream>

#include "config_extraction.h"

inline std::shared_ptr<spdlog::logger> logger;

void initLogger();
void log(const std::string& text, spdlog::level::level_enum lvl = spdlog::level::info);

#endif