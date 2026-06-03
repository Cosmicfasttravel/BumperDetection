#ifndef BUMPERDETECTION_CONFIG_EXTRACTION_H
#define BUMPERDETECTION_CONFIG_EXTRACTION_H
#include "config_types.h"

namespace config {

    const Config &getRef();
    bool tryUpdate();
    uint64_t getVersion();

}

#endif // BUMPERDETECTION_CONFIG_EXTRACTION_H