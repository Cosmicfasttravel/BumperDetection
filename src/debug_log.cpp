#include "debug_log.h"
#include "config_extraction.h"
#include <chrono>

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
}