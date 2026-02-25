cmake_minimum_required(VERSION 3.21)
cmake_policy(SET CMP0146 OLD)
project(BumperDetection)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED TRUE)

# Point to CUDA-enabled OpenCV
set(OpenCV_DIR "C:/opencv_cuda_install")
find_package(OpenCV REQUIRED core imgproc imgcodecs text video videoio highgui)
find_package(Leptonica CONFIG REQUIRED)
find_package(Tesseract CONFIG REQUIRED)
find_package(CUDA REQUIRED)

# Check if OpenCV has CUDA support
if(OpenCV_CUDA_VERSION)
    message(STATUS "✓ OpenCV built with CUDA: ${OpenCV_CUDA_VERSION}")
    add_definitions(-DHAVE_OPENCV_CUDNN)
else()
    message(WARNING "✗ OpenCV NOT built with CUDA support")
endif()

# ONNX Runtime
set(ONNXRUNTIME_DIR "C:/onnxruntime")
if(EXISTS ${ONNXRUNTIME_DIR})
    include_directories(${ONNXRUNTIME_DIR}/include)
    set(ONNX_LIBS "${ONNXRUNTIME_DIR}/lib/onnxruntime.lib")
    set(ONNX_DLL "${ONNXRUNTIME_DIR}/lib/onnxruntime.dll")
else()
    set(ONNX_LIBS "")
    set(ONNX_DLL "")
endif()

# Debug output
message(STATUS "=====================================")
message(STATUS "Build Configuration:")
message(STATUS "=====================================")
message(STATUS "OpenCV_VERSION: ${OpenCV_VERSION}")
message(STATUS "OpenCV_DIR: ${OpenCV_DIR}")
if(OpenCV_CUDA_VERSION)
    message(STATUS "CUDA: YES (${OpenCV_CUDA_VERSION})")
else()
    message(STATUS "CUDA: NO")
endif()
message(STATUS "=====================================")

include_directories(${OpenCV_INCLUDE_DIRS})

add_executable(BumperDetection
        C:/Users/marcu/CLionProjects/robotvisiontest/src/main.cpp
        C:/Users/marcu/CLionProjects/robotvisiontest/src/setupGPU.cpp
        C:/Users/marcu/CLionProjects/robotvisiontest/src/setupGPU.h
        C:/Users/marcu/CLionProjects/robotvisiontest/src/robotTracking.cpp
        C:/Users/marcu/CLionProjects/robotvisiontest/src/robotTracking.h
        C:/Users/marcu/CLionProjects/robotvisiontest/src/kalmanFilter.cpp
        C:/Users/marcu/CLionProjects/robotvisiontest/src/kalmanFilter.h
        C:/Users/marcu/CLionProjects/robotvisiontest/src/topDownView.cpp
        C:/Users/marcu/CLionProjects/robotvisiontest/src/topDownView.h
)

set(TENSORRT_ROOT "C:/TensorRT-10.15.1.29")

target_include_directories(BumperDetection PRIVATE
        ${TENSORRT_ROOT}/include
)

include_directories(${CUDA_INCLUDE_DIRS})

target_link_directories(BumperDetection PRIVATE
        ${TENSORRT_ROOT}/lib
)

target_link_libraries(BumperDetection
        ${OpenCV_LIBS}
        opencv_core
        opencv_imgproc
        opencv_imgcodecs
        opencv_video
        opencv_videoio
        opencv_highgui
        opencv_text
        Tesseract::libtesseract
        ${ONNX_LIBS}
        nvinfer_10
        nvonnxparser_10
        nvinfer_plugin_10
        ${CUDA_LIBRARIES}
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

    # Copy CUDA DLLs
    if(DEFINED ENV{CUDA_PATH})
        file(GLOB CUDA_DLLS "$ENV{CUDA_PATH}/bin/cudart*.dll")
        file(GLOB CUDNN_DLLS "C:/Program Files/NVIDIA/CUDNN/v9.17/bin/13.1/*.dll")

        foreach(dll ${CUDA_DLLS})
            add_custom_command(TARGET BumperDetection POST_BUILD
                    COMMAND ${CMAKE_COMMAND} -E copy_if_different
                    "${dll}"
                    $<TARGET_FILE_DIR:BumperDetection>
            )
        endforeach()

        foreach(dll ${CUDNN_DLLS})
            add_custom_command(TARGET BumperDetection POST_BUILD
                    COMMAND ${CMAKE_COMMAND} -E copy_if_different
                    "${dll}"
                    $<TARGET_FILE_DIR:BumperDetection>
            )
        endforeach()
    endif()
endif()