cmake_minimum_required(VERSION 3.16)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

project(BumperDetection)

cmake_policy(SET CMP0135 NEW)

include(FetchContent)

unset(BUILD_CORE CACHE)
unset(BUILD_TESS CACHE)
unset(BUILD_NT CACHE)
unset(BUILD_TRACK CACHE)
unset(BUILD_WEBUI CACHE)
unset(BUILD_LOGGER CACHE)

option(BUILD_CORE "Build the core engine" ON)
option(BUILD_TESS "Build the optional tesseract module" ON)
option(BUILD_NT "Build the optional network tables module" OFF)
option(BUILD_TRACK "Build the optional tracking module" OFF)
option(BUILD_WEBUI "Build the optional WebUI module" OFF)
option(BUILD_LOGGER "Build the optional Logger module" ON)

unset(INFERENCE_BACKEND CACHE)
set(INFERENCE_BACKEND "CPU" CACHE STRING "Inference backend") # Must type exactly the backend
set_property(CACHE INFERENCE_BACKEND PROPERTY STRINGS
        CPU
        NVIDIA_GPU
        RKNN_NPU
)

message(STATUS "----------------------------------------")
message(STATUS " Configuration Summary:")
message(STATUS "   Core Module:    ${BUILD_CORE}")
message(STATUS "   Tesseract Module:   ${BUILD_TESS}")
message(STATUS "   Network Tables Module: ${BUILD_NT}")
message(STATUS "   Tracking Module: ${BUILD_TRACK}")
message(STATUS "   WebUI Module: ${BUILD_WEBUI}")
message(STATUS "   Logger Module: ${BUILD_LOGGER}")
message(STATUS "----------------------------------------")
message(STATUS " Target device: " ${INFERENCE_BACKEND})
message(STATUS "----------------------------------------")


if (BUILD_CORE)
    message(STATUS " Building Core Module")

    FetchContent_Declare(json URL https://github.com/nlohmann/json/releases/download/v3.12.0/json.tar.xz)
    FetchContent_MakeAvailable(json)

    find_package(OpenCV REQUIRED COMPONENTS core imgproc highgui videoio dnn video)

    message(STATUS " Built Core Module")

endif ()

if (BUILD_TESS)
    message(STATUS " Building Tesseract Module")

    if (CMAKE_SYSTEM_NAME STREQUAL "Linux")
        find_package(PkgConfig REQUIRED)
        pkg_check_modules(LEPT REQUIRED lept)
        pkg_check_modules(TESS REQUIRED tesseract)
    endif ()

    if (WIN32)
        set(CMAKE_WINDOWS_EXPORT_ALL_SYMBOLS ON)
        find_package(Tesseract CONFIG REQUIRED)
        find_package(Leptonica CONFIG REQUIRED)
    endif ()

    message(STATUS " Built Tesseract Module")

endif ()

if (BUILD_NT)
    message(STATUS " Building Network Tables Module")

    message(STATUS " Built Network Tables Module")

endif ()

if (BUILD_TRACK)
    message(STATUS " Building Tracking Module")

    FetchContent_Declare(
            eigen
            GIT_REPOSITORY https://gitlab.com/libeigen/eigen.git
            GIT_TAG 3.4.0
    )
    set(EIGEN_BUILD_DOC OFF CACHE BOOL "" FORCE)
    set(BUILD_TESTING OFF CACHE BOOL "" FORCE)
    set(EIGEN_BUILD_PKGCONFIG OFF CACHE BOOL "" FORCE)
    FetchContent_MakeAvailable(eigen)

    list(APPEND CMAKE_PREFIX_PATH "../yaml-cpp/install")
    set(YAML_CPP_INCLUDE_DIR "../yaml-cpp/install/include/")

    set(YAML_CPP_BUILD_TESTS OFF CACHE BOOL "" FORCE)
    set(YAML_CPP_BUILD_TOOLS OFF CACHE BOOL "" FORCE)
    set(YAML_CPP_INSTALL OFF CACHE BOOL "" FORCE)

    find_package(yaml-cpp REQUIRED)
    set(MOTCPP_INCLUDE_DIR "../motcpp/include")
    set(MOTCPP_LIB "../motcpp/build/Release/")

    message(STATUS " Built Track Module")

endif ()


if (BUILD_WEBUI)
    message(STATUS " Building WebUI Module")

    message(STATUS " Built WebUI Module")
endif ()

if (BUILD_LOGGER)
    message(STATUS " Building Logger Module")

    FetchContent_Declare(spdlog URL https://github.com/gabime/spdlog/archive/refs/tags/v1.17.0.tar.gz)
    FetchContent_MakeAvailable(spdlog)

    message(STATUS " Built Logger Module")

endif ()

message(STATUS " Adding library and linking...")

add_library(BumperDetection STATIC)

target_compile_definitions(
        BumperDetection
        PUBLIC
        INFERENCE_BACKEND="${INFERENCE_BACKEND}"
)

if (BUILD_CORE)
    target_sources(BumperDetection PRIVATE src/core/kalman/kalman_filter.cpp)
    target_sources(BumperDetection PRIVATE src/core/config/config_extraction.cpp)
    target_sources(BumperDetection PRIVATE src/core/threading/thread_manager.cpp)
    target_sources(BumperDetection PRIVATE src/global/build_info.h)
    target_sources(BumperDetection PRIVATE src/core/capture/camera/camera_capture.cpp)
    target_sources(BumperDetection PRIVATE src/core/capture/video/video_capture.cpp)

    target_include_directories(BumperDetection PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/src)
    target_include_directories(BumperDetection PRIVATE ${OpenCV_INCLUDE_DIRS})

    target_compile_definitions(BumperDetection PUBLIC BUILD_CORE)
endif ()
if (BUILD_TESS)
    target_sources(BumperDetection PRIVATE src/tesseract/tesseract_engine.cpp)

    target_compile_definitions(BumperDetection PUBLIC BUILD_TESS)
endif ()
if (BUILD_NT)
    target_sources(BumperDetection PRIVATE src/network-tables)

    target_compile_definitions(BumperDetection PUBLIC BUILD_NT)
endif ()
if (BUILD_TRACK)
    target_sources(BumperDetection PRIVATE src/tracking)
    target_include_directories(BumperDetection PRIVATE ${MOTCPP_INCLUDE_DIR})

    target_compile_definitions(BumperDetection PUBLIC BUILD_TRACK)
endif ()
if (BUILD_WEBUI)
    target_sources(BumperDetection PRIVATE src/webui)

    target_compile_definitions(BumperDetection PUBLIC BUILD_WEBUI)
endif ()
if (BUILD_LOGGER)
    target_sources(BumperDetection PRIVATE src/log/logger.cpp)

    target_compile_definitions(BumperDetection PUBLIC BUILD_LOGGER)
endif ()

if (CMAKE_SYSTEM_NAME STREQUAL "Linux")
    target_include_directories(BumperDetection PUBLIC
            ${LEPT_INCLUDE_DIRS}
            ${TESS_INCLUDE_DIRS}
    )
endif ()

if (BUILD_CORE)
    target_link_libraries(BumperDetection PUBLIC
            ${OpenCV_LIBS}
            nlohmann_json::nlohmann_json
    )
endif ()

if (BUILD_TRACK)

    target_link_libraries(BumperDetection PUBLIC Eigen3::Eigen)
    target_include_directories(BumperDetection PUBLIC ${MOTCPP_INCLUDE_DIR})

endif ()

if (BUILD_LOGGER)
    target_link_libraries(BumperDetection PUBLIC spdlog::spdlog)
endif ()

if (BUILD_TESS)
    if (CMAKE_SYSTEM_NAME STREQUAL "Linux")

        target_link_libraries(BumperDetection PUBLIC
                ${TESS_LIBRARIES}
                ${LEPT_LIBRARIES}
        )

    endif ()

    if (WIN32)
        target_include_directories(BumperDetection PUBLIC
                ${Tesseract_INCLUDE_DIRS}
        )
        target_link_libraries(BumperDetection PUBLIC
                Tesseract::libtesseract
        )
    endif ()
endif ()

if (INFERENCE_BACKEND STREQUAL "RKNN_NPU")
    target_link_libraries(BumperDetection PUBLIC rknnrt)
endif ()

add_executable(run_yolo src/exec/main.cpp)

target_link_libraries(run_yolo PRIVATE BumperDetection)

target_include_directories(run_yolo PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)

if(MSVC)
    target_compile_options(BumperDetection PRIVATE /W4)
else()
    target_compile_options(BumperDetection PRIVATE -Wall -Wextra)
endif()