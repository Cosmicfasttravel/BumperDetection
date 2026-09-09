#ifndef BUMPERDETECTION_COMPONENTS_H
#define BUMPERDETECTION_COMPONENTS_H

namespace build_info {
    inline constexpr bool core =
#ifdef BUILD_CORE
            true;
#else
    false;
#endif

    inline constexpr bool tess =
#ifdef BUILD_TESS
            true;
#else
            false;
#endif

    inline constexpr bool track =
#ifdef BUILD_TRACK
            true;
#else
            false;
#endif

    inline constexpr bool logger =
#ifdef BUILD_LOGGER
            true;
#else
    false;
#endif

}

#endif //BUMPERDETECTION_COMPONENTS_H
