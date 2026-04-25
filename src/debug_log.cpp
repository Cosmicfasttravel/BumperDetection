#include "debug_log.h"
#include "config_extraction.h"
#include <chrono>

Config config;

void initLogger()
{
    try
    {
        logger = spdlog::basic_logger_mt("debug_log", "logs/debug-log.txt"); // config for location
    }
    catch (const spdlog::spdlog_ex &ex)
    {
        std::cout << "Log init failed: " << ex.what() << std::endl;
    }
    spdlog::flush_every(std::chrono::seconds(3));

    config = getConfig();
}


void log(const std::string& text, const spdlog::level::level_enum lvl) {
    if (lvl == spdlog::level::debug) if (config.modes.logging) logger->debug(text);
    if (lvl == spdlog::level::info) if (config.modes.logging) logger->info(text);
    if (lvl == spdlog::level::warn) if (config.modes.logging) logger->warn(text);
    if (lvl == spdlog::level::err) if (config.modes.logging) logger->error(text);
    if (lvl == spdlog::level::critical) if (config.modes.logging) logger->critical(text);
}