#ifndef BUMPERDETECTION_DEBUG_LOG_H
#define BUMPERDETECTION_DEBUG_LOG_H

#include "spdlog/sinks/basic_file_sink.h"

namespace logging {
    void write(const std::string& text, spdlog::level::level_enum lvl = spdlog::level::info);
}

#endif