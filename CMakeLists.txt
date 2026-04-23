cmake_minimum_required(VERSION 3.16)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

project(BumperDetection)

set(CMAKE_POSITION_INDEPENDENT_CODE ON)

include(FetchContent)

FetchContent_Declare(json URL https://github.com/nlohmann/json/releases/download/v3.12.0/json.tar.xz)
FetchContent_MakeAvailable(json)

FetchContent_Declare(spdlog URL https://github.com/gabime/spdlog/archive/refs/tags/v1.17.0.tar.gz)
FetchContent_MakeAvailable(spdlog)

FetchContent_Declare(
        eigen
        GIT_REPOSITORY https://gitlab.com/libeigen/eigen.git
        GIT_TAG 3.4.0
)
set(EIGEN_BUILD_DOC OFF CACHE BOOL "" FORCE)
set(BUILD_TESTING OFF CACHE BOOL "" FORCE)
set(EIGEN_BUILD_PKGCONFIG OFF CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(eigen)

find_package(OpenCV REQUIRED COMPONENTS core imgproc highgui videoio dnn video)

list(APPEND CMAKE_PREFIX_PATH "../yaml-cpp/install")

set(YAML_CPP_INCLUDE_DIR "../yaml-cpp/install/include/")

set(YAML_CPP_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(YAML_CPP_BUILD_TOOLS OFF CACHE BOOL "" FORCE)
set(YAML_CPP_INSTALL OFF CACHE BOOL "" FORCE)

find_package(yaml-cpp REQUIRED)
set(MOTCPP_INCLUDE_DIR "../motcpp/include")
set(MOTCPP_LIB "../motcpp/build/Release/motcpp.lib")

file(GLOB_RECURSE WPILIB_INCLUDE_DIRS LIST_DIRECTORIES true
    ../allwpilib/*/src/main/native/include
    ../allwpilib/*/src/main/native/thirdparty/*/include
)

file(GLOB WPILIB_LIBS
    ../allwpilib/*/build/libs/*/shared/linuxarm64/release/*.so
)

if (UNIX)
    find_package(PkgConfig REQUIRED)
    pkg_check_modules(LEPT REQUIRED lept)
    pkg_check_modules(TESS REQUIRED tesseract)
endif ()

if (WIN32)
    set(CMAKE_WINDOWS_EXPORT_ALL_SYMBOLS ON)
    find_package(Tesseract CONFIG REQUIRED)
    find_package(Leptonica CONFIG REQUIRED)
endif ()

add_library(BumperDetection SHARED
        src/bumper_detector.cpp
        src/analyze_detections.cpp
        src/kalman_filter.cpp
        src/send_data.cpp
        src/config_extraction.cpp
        src/debug_log.cpp
        src/debug_log.h
)

target_include_directories(BumperDetection PRIVATE ${OpenCV_INCLUDE_DIRS} ${MOTCPP_INCLUDE_DIR} ${WPILIB_INCLUDE_DIRS})

if (UNIX)
    target_include_directories(BumperDetection
            PRIVATE
            ${LEPT_INCLUDE_DIRS}
            ${TESS_INCLUDE_DIRS}
    )
endif ()

target_link_libraries(BumperDetection
        ${OpenCV_LIBS}
        nlohmann_json::nlohmann_json
        spdlog::spdlog
        Eigen3::Eigen
        ${WPILIB_LIBS}
)

if (UNIX)
    target_link_libraries(BumperDetection
            rknnrt
            ${TESS_LIBRARIES}
            ${LEPT_LIBRARIES}
    )
endif ()

if (WIN32)
    target_include_directories(BumperDetection PRIVATE
            ${Tesseract_INCLUDE_DIRS}
    )
    target_link_libraries(BumperDetection
            Tesseract::libtesseract
    )
endif()

add_executable(run_yolo src/main.cpp)

target_include_directories(run_yolo PRIVATE ${MOTCPP_INCLUDE_DIR} ${WPILIB_INCLUDE_DIRS})

target_link_libraries(run_yolo
        BumperDetection 
        ${OpenCV_LIBS}
        nlohmann_json::nlohmann_json
        spdlog::spdlog
        Eigen3::Eigen
        ${WPILIB_LIBS}
)

if(WIN32)
        target_link_libraries(run_yolo
        Tesseract::libtesseract
        )
endif()

if (UNIX)
    target_link_libraries(run_yolo
            ${TESS_LIBRARIES}
            ${LEPT_LIBRARIES}
            rknnrt
    )
endif ()
