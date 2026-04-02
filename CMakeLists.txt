cmake_minimum_required(VERSION 3.16)
project(BumperDetection)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

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
        src/top_down_view.cpp
        src/send_data.cpp
        src/receive_data.cpp
        src/config_extraction.cpp
        src/log_to_file.cpp
)

target_link_libraries(BumperDetection
        ${OpenCV_LIBS}
)

if (UNIX)
    target_link_libraries(BumperDetection
            rknnrt
            ${TESS_INCLUDE_DIRS}
            ${LEPT_INCLUDE_DIRS}
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
        Tesseract::libtesseract
        ${OpenCV_LIBS}
)

if (UNIX)
    target_link_libraries(run_yolo
            ntcore
            wpiutil
            wpinet
    )
endif ()