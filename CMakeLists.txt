cmake_minimum_required(VERSION 3.16)
project(BumperDetection)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

include(FetchContent)

FetchContent_Declare(json URL https://github.com/nlohmann/json/releases/download/v3.12.0/json.tar.xz)
FetchContent_MakeAvailable(json)

find_package(OpenCV REQUIRED COMPONENTS core imgproc highgui videoio dnn video)

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

include_directories(${OpenCV_INCLUDE_DIRS})

if (UNIX)
    include_directories(
            ${LEPT_INCLUDE_DIRS}
            ${TESS_INCLUDE_DIRS}
    )
endif ()

add_library(BumperDetection SHARED
        src/bumper_detector.cpp
        src/analyze_detections.cpp
        src/kalman_filter.cpp
        src/send_data.cpp
        src/config_extraction.cpp
        src/thread_manager.cpp
)

target_link_libraries(BumperDetection
        ${OpenCV_LIBS}
        nlohmann_json::nlohmann_json
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

target_link_libraries(run_yolo
        BumperDetection 
        ${OpenCV_LIBS}
        nlohmann_json::nlohmann_json
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
            ntcore
            wpiutil
            wpinet
            rknnrt
    )
endif ()