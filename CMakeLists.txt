cmake_minimum_required(VERSION 3.21)
project(BumperDetection)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED TRUE)

# Point to CUDA-enabled OpenCV
set(OpenCV_DIR "C:/opencv_cuda_install")
find_package(OpenCV REQUIRED core imgproc imgcodecs text video videoio highgui)
find_package(Leptonica CONFIG REQUIRED)
find_package(Tesseract CONFIG REQUIRED)

# Check if OpenCV has CUDA support
if(OpenCV_CUDA_VERSION)
    message(STATUS "✓ OpenCV built with CUDA: ${OpenCV_CUDA_VERSION}")
    add_definitions(-DHAVE_OPENCV_CUDNN)
else()
    message(WARNING "✗ OpenCV NOT built with CUDA support")
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
        opencv_text
        Tesseract::libtesseract
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