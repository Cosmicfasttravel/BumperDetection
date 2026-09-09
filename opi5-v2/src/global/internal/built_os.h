#ifndef BUMPERDETECTION_BUILT_OS_H
#define BUMPERDETECTION_BUILT_OS_H

namespace build_info {
    inline constexpr bool is_windows =
#ifdef _WIN32
            true;
#else
    false;
#endif

    inline constexpr bool is_linux =
#ifdef __linux__
            true;
#else
            false;
#endif

    inline constexpr bool is_macos =
#ifdef __APPLE__
            true;
#else
            false;
#endif
}

#endif //BUMPERDETECTION_BUILT_OS_H
