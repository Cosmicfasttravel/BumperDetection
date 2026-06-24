#include "./camera_capture.h"

#include "core/config/config_extraction.h"
#include "global/internal/built_os.h"
#include "log/logger.h"
#include "../video/video_capture.h"

camera_capture::camera_capture() = default;

camera_capture::~camera_capture() {
    shutdownCaptureComponent();
}

void camera_capture::initializeCaptureComponent() {
    configureCaptureComponent();
    camThread = std::thread(&camera_capture::capture, this, std::ref(cap));
}

void camera_capture::shutdownCaptureComponent() {
    capturing = false;
    if (!videoMode && camThread.joinable()) camThread.join();
    cap.release();
}

void camera_capture::configureCaptureComponent() {
    std::string videoPath = config::getLatestCopy().input_paths.video_path;

    if (videoPath.empty()) {
        logging::write("VideoPath is empty, defaulting to camera mode");
        if (build_info::is_windows) cap.open(0, cv::CAP_DSHOW);
        else cap.open(0, cv::CAP_V4L2);
        videoMode = false;
    } else {
        logging::write("VideoPath is filled, switching to video mode");
        cap.open(videoPath);
        videoMode = true;
    }

    const auto config = config::getLatestCopy();
    cap.set(cv::CAP_PROP_FOURCC, cv::VideoWriter::fourcc('M', 'J', 'P', 'G'));
    cap.set(cv::CAP_PROP_FRAME_WIDTH, config.screen.width);
    cap.set(cv::CAP_PROP_FRAME_HEIGHT, config.screen.height);
    cap.set(cv::CAP_PROP_FPS, 60);
    cap.set(cv::CAP_PROP_BUFFERSIZE, 1);

    if (config::checkConfigVersion(config)) {
        cap.set(cv::CAP_PROP_BRIGHTNESS, config.camera.brightness);
        cap.set(cv::CAP_PROP_CONTRAST, config.camera.contrast);
        cap.set(cv::CAP_PROP_HUE, config.camera.hue);
        cap.set(cv::CAP_PROP_SATURATION, config.camera.saturation);
        cap.set(cv::CAP_PROP_GAIN, config.camera.gain);
        cap.set(cv::CAP_PROP_EXPOSURE, config.camera.exposure);
        cap.set(cv::CAP_PROP_WB_TEMPERATURE, config.camera.temperature);
        cap.set(cv::CAP_PROP_AUTO_WB, config.camera.temperature);
        cap.set(cv::CAP_PROP_FRAME_WIDTH, config.screen.width);
        cap.set(cv::CAP_PROP_FRAME_HEIGHT, config.screen.height);
    }
}

cv::Mat camera_capture::retrieveLatestFrame() {
    cv::Mat frame; {
        std::lock_guard<std::mutex> lock(frameMutex);
        if (currentFrame.empty()) return frame;
        frame = currentFrame.clone();
    }
    return frame;
}

void camera_capture::capture(cv::VideoCapture &capture) {
    cv::Mat frame;

    while (capturing) {
        if (!capture.isOpened()) return;
        capture.read(frame);

        if (!frame.empty()) {
            {
                std::lock_guard<std::mutex> lock(frameMutex);
                currentFrame = frame.clone();
            }
            video_capture::pushFrame(currentFrame);
        }
    }
}
