cmake_minimum_required(VERSION 3.21)
project(BumperDetection)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED TRUE)

set(OpenCV_DIR "C:/opencv_opencl_install/build")
find_package(OpenCV REQUIRED)
find_package(Leptonica CONFIG REQUIRED)
find_package(Tesseract CONFIG REQUIRED)

# ONNX Runtime (if you're still using it)
set(ONNXRUNTIME_DIR "C:/onnxruntime")
if(EXISTS ${ONNXRUNTIME_DIR})
    include_directories(${ONNXRUNTIME_DIR}/include)
    set(ONNX_LIBS "${ONNXRUNTIME_DIR}/lib/onnxruntime.lib")
    set(ONNX_DLL "${ONNXRUNTIME_DIR}/lib/onnxruntime.dll")
else()
    set(ONNX_LIBS "")
    set(ONNX_DLL "")
endif()

include_directories(${OpenCV_INCLUDE_DIRS})

add_executable(BumperDetection
        C:/Users/marcu/CLionProjects/robotvisiontest/src/yolomodelgpu.cpp
        C:/Users/marcu/CLionProjects/robotvisiontest/src/GPU.cpp
        C:/Users/marcu/CLionProjects/robotvisiontest/src/GPU.h
        C:/Users/marcu/CLionProjects/robotvisiontest/src/edgeDetection.cpp
        C:/Users/marcu/CLionProjects/robotvisiontest/src/edgeDetection.h
        C:/Users/marcu/CLionProjects/robotvisiontest/src/kalmanFilter.cpp
        C:/Users/marcu/CLionProjects/robotvisiontest/src/kalmanFilter.h
        C:/Users/marcu/CLionProjects/robotvisiontest/src/topDownView.cpp
        C:/Users/marcu/CLionProjects/robotvisiontest/src/topDownView.h
)

target_link_libraries(BumperDetection
        ${OpenCV_LIBS}
        opencv_core
        opencv_imgproc
        opencv_imgcodecs
        opencv_video
        opencv_videoio
        opencv_highgui
        Tesseract::libtesseract
        ${ONNX_LIBS}
)

# Copy OpenCV DLLs
if(WIN32)
    # Get OpenCV bin directory
    get_filename_component(OpenCV_BIN_DIR "${OpenCV_DIR}/x64/vc17/bin" ABSOLUTE)

    if(NOT EXISTS ${OpenCV_BIN_DIR})
        set(OpenCV_BIN_DIR "${OpenCV_DIR}/bin")
    endif()

    message(STATUS "OpenCV DLL directory: ${OpenCV_BIN_DIR}")

    # Copy OpenCV DLLs
    file(GLOB OPENCV_DLLS "${OpenCV_BIN_DIR}/*.dll")
    foreach(dll ${OPENCV_DLLS})
        add_custom_command(TARGET BumperDetection POST_BUILD
                COMMAND ${CMAKE_COMMAND} -E copy_if_different
                "${dll}"
                $<TARGET_FILE_DIR:BumperDetection>
        )
    endforeach()

    # Copy ONNX DLL
    if(EXISTS ${ONNX_DLL})
        add_custom_command(TARGET BumperDetection POST_BUILD
                COMMAND ${CMAKE_COMMAND} -E copy_if_different
                "${ONNX_DLL}"
                $<TARGET_FILE_DIR:BumperDetection>
        )
    endif()
endif()