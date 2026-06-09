#include "log/logger.h"

#include <chrono>
#include <iostream>

#include "spdlog/spdlog.h"

#include "global/build_info.h"

inline std::shared_ptr<spdlog::logger> logger;

std::once_flag loggerInitFlag;
void initLogger() {
    try {
        logger = spdlog::basic_logger_mt("debug_log", "logs/debug-log.txt");
    } catch (const spdlog::spdlog_ex &ex) {
        std::cout << "Log init failed: " << ex.what() << std::endl;
    }
    spdlog::flush_every(std::chrono::seconds(3));
}

void ensureLogger() {
    std::call_once(loggerInitFlag, [] {
        if constexpr (!build_info::logger) spdlog::set_level(spdlog::level::off);
        initLogger();
    });
}

void write(const std::string &text, const spdlog::level::level_enum lvl) {
    ensureLogger();

    if (!logger) return;

    if (text.empty()) return;

    logger->log(lvl, text);
}
