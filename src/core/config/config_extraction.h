#ifndef BUMPERDETECTION_CONFIG_EXTRACTION_H
#define BUMPERDETECTION_CONFIG_EXTRACTION_H

#include "core/config/config_types.h"

namespace config {

    Config getLatestCopy();
    bool tryUpdate();
    bool checkConfigVersion(const Config& config);

}

#endif // BUMPERDETECTION_CONFIG_EXTRACTION_H