#include "video_capture.h"

#include "core/config/config_extraction.h"
#include "global/internal/built_os.h"

video_capture::video_capture() = default;

video_capture::~video_capture() {
    shutdownVideoCapture();
}

void video_capture::initializeVideoCapture() {
    if (!captureMode) return;
    configureVideoCapture();

    vidThread = std::thread(&video_capture::writeToFile, this);
}

void video_capture::configureVideoCapture() {
    if (!captureMode) return;

    if constexpr (build_info::is_linux) codec = cv::VideoWriter::fourcc('a', 'v', 'c', '1');
    else codec = cv::VideoWriter::fourcc('m', 'p', '4', 'v');

    const Config config = config::getLatestCopy();

    double fpsVideo = config.screen.target_fps;
    if (fpsVideo <= 0)
        fpsVideo = 15.0;

    int frame_width = static_cast<int>(config.screen.width);
    int frame_height = static_cast<int>(config.screen.height);

    std::string filename = "./output/output.mp4";
    writer.open(filename, codec, fpsVideo, cv::Size(frame_width, frame_height), true);
}

void video_capture::pushFrame(const cv::Mat &frame) { {
        std::lock_guard lock(videoMutex);
        frameQueue.push(frame.clone());

        if (frameQueue.size() > 100)
            frameQueue.pop();
    }

    frameCv.notify_one();
}

void video_capture::writeToFile() {
    while (true) {
        try {
            std::unique_lock lock(videoMutex);

            frameCv.wait(lock, [&] {
                return !frameQueue.empty() || !running;
            });

            if (!running && frameQueue.empty())
                break;

            cv::Mat frame = std::move(frameQueue.front());
            frameQueue.pop();

            lock.unlock();

            writer.write(frame);

        } catch (std::exception &e) {
            std::cerr << e.what() << std::endl;

        } catch (...) {
            std::cerr << "Unknown exception" << std::endl;
        }
    }
}

void video_capture::shutdownVideoCapture() { {
        std::lock_guard lock(videoMutex);
        running = false;
    }

    frameCv.notify_all();

    if (vidThread.joinable())
        vidThread.join();

    writer.release();
}
