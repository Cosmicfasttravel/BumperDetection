cmake_minimum_required(VERSION 3.21)
project(BumperDetection)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED TRUE)

# OpenCV
set(OpenCV_DIR "C:/opencv/build")
find_package(OpenCV REQUIRED)

# ONNX Runtime
set(ONNXRUNTIME_DIR "C:/onnxruntime")
include_directories(${ONNXRUNTIME_DIR}/include)

# Debug output
message(STATUS "OpenCV_DIR: ${OpenCV_DIR}")
message(STATUS "OpenCV_INCLUDE_DIRS: ${OpenCV_INCLUDE_DIRS}")
message(STATUS "OpenCV_LIBS: ${OpenCV_LIBS}")
message(STATUS "OpenCV_VERSION: ${OpenCV_VERSION}")

add_executable(BumperDetection main.cpp
        "yolomodel.cpp")

# Only link opencv_world (it contains all modules)
target_link_libraries(BumperDetection
        opencv_world
        "${ONNXRUNTIME_DIR}/lib/onnxruntime.lib"
)

# Copy necessary DLLs to build directory
add_custom_command(TARGET BumperDetection POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
        "${OpenCV_DIR}/x64/vc16/bin/$<IF:$<CONFIG:Debug>,opencv_world4120d.dll,opencv_world4120.dll>"
        $<TARGET_FILE_DIR:BumperDetection>
)

add_custom_command(TARGET BumperDetection POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
        "${ONNXRUNTIME_DIR}/lib/onnxruntime.dll"
        $<TARGET_FILE_DIR:BumperDetection>
)