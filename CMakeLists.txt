cmake_minimum_required(VERSION 3.16)
project(BumperDetection)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

find_package(OpenCV REQUIRED COMPONENTS core imgproc highgui videoio dnn video)
find_package(PkgConfig REQUIRED)

pkg_check_modules(LEPT REQUIRED lept)
pkg_check_modules(TESS REQUIRED tesseract)

include_directories(
	${OpenCV_INCLUDE_DIRS}
	${LEPT_INCLUDE_DIRS}
	${TESS_INCLUDE_DIRS})

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
        ${LEPT_LIBRARIES}
        ${TESS_LIBRARIES}
        rknnrt
)

add_executable(run_yolo src/main.cpp)

target_link_libraries(run_yolo BumperDetection 
${OpenCV_LIBS}
${LEPT_LIBRARIES}
${TESS_LIBRARIES}
ntcore
wpiutil
wpinet
)
