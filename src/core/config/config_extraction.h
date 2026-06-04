#ifndef BUMPERDETECTION_CONFIG_EXTRACTION_H
#define BUMPERDETECTION_CONFIG_EXTRACTION_H

#include "config_types.h"

namespace config {

    Config getLatestCopy();
    bool tryUpdate();
    uint64_t getVersion();
    bool checkConfigVersion(const Config& config);

}

#endif // BUMPERDETECTION_CONFIG_EXTRACTION_H