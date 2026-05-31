#include "logger.h"
#include "config_extraction.h"
#include <chrono>
#include <iostream>

#include "spdlog/spdlog.h"

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
        initLogger();
    });
}

void write(const std::string &text, const spdlog::level::level_enum lvl) {

    ensureLogger();

    if (!config::getRef().modes.logging) return;
    if (text.empty()) return;

    logger->log(lvl, text);
}
