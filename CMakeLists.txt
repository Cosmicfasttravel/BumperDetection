cmake_minimum_required(VERSION 3.16)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

project(BumperDetection)

set(CMAKE_POSITION_INDEPENDENT_CODE ON)

include(FetchContent)

option(BUILD_CORE "Build the core engine" ON)
option(BUILD_TESS "Build the optional tesseract module" OFF)
option(BUILD_NT "Build the optional network tables module" OFF)
option(BUILD_TRACK "Build the optional WebUI module" OFF)
option(BUILD_WEBUI "Build the optional WebUI module" OFF)
option(BUILD_LOGGER "Build the optional Logger module" OFF)

message(STATUS "----------------------------------------")
message(STATUS " Configuration Summary:")
message(STATUS "   Core Module:    ${BUILD_CORE}")
message(STATUS "   Tesseract Module:   ${BUILD_TESS}")
message(STATUS "   Network Tables Module: ${BUILD_NT}")
message(STATUS "   Tracking Module: ${BUILD_TRACK}")
message(STATUS "   WebUI Module: ${BUILD_WEBUI}")
message(STATUS "   Logger Module: ${BUILD_LOGGER}")
message(STATUS "----------------------------------------")

if(BUILD_CORE)
    message(STATUS "----------------------------------------")
    message(STATUS " Building Core Module")
    message(STATUS "----------------------------------------")

    FetchContent_Declare(json URL https://github.com/nlohmann/json/releases/download/v3.12.0/json.tar.xz)
    FetchContent_MakeAvailable(json)

    find_package(OpenCV REQUIRED COMPONENTS core imgproc highgui videoio dnn video)

endif ()

if(BUILD_TESS)
    message(STATUS "----------------------------------------")
    message(STATUS " Building Tesseract Module")
    message(STATUS "----------------------------------------")

endif ()

if(BUILD_NT)
    message(STATUS "----------------------------------------")
    message(STATUS " Building Network Tables Module")
    message(STATUS "----------------------------------------")

endif ()

if(BUILD_TRACK)
    message(STATUS "----------------------------------------")
    message(STATUS " Building Tracking Module")
    message(STATUS "----------------------------------------")

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
    set(MOTCPP_LIB "../motcpp/build/Release/motcpp.lib")

endif ()


if(BUILD_WEBUI)
    message(STATUS "----------------------------------------")
    message(STATUS " Building WebUI Module")
    message(STATUS "----------------------------------------")

endif ()

if(BUILD_LOGGER)
    message(STATUS "----------------------------------------")
    message(STATUS " Building Logger Module")
    message(STATUS "----------------------------------------")

    FetchContent_Declare(spdlog URL https://github.com/gabime/spdlog/archive/refs/tags/v1.17.0.tar.gz)
    FetchContent_MakeAvailable(spdlog)

endif ()

