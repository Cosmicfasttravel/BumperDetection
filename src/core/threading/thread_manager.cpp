#include "core/threading/thread_manager.h"

ThreadManager::ThreadManager(int thread_count) {
    Num_Threads = std::clamp(
        thread_count,
        1,
        static_cast<int>(std::thread::hardware_concurrency())
    );

    for (int i = 0; i < *Num_Threads; i++) {
        Pool.emplace_back(&ThreadManager::workerLoop, this);
    }
}

ThreadManager::~ThreadManager() {
    shutdown();
}


void ThreadManager::workerLoop() {
    while (true) {
        std::function<void()> Job; {
            std::unique_lock<std::mutex> lock(Queue_Mutex);

            condition.wait(lock, [this] {
                return stop || !Queue.empty();
            });
            if (stop && Queue.empty())
                return;

            Job = Queue.front();
            Queue.pop();
        }
        Job();
    }
}

void ThreadManager::addJob(const std::function<void()> &New_Job) { {
        std::unique_lock<std::mutex> lock(Queue_Mutex);
        if (stop)
            return;

        Queue.push(New_Job);
    }
    condition.notify_one();
}

void ThreadManager::shutdown() {
    if (stop) return; {
        std::unique_lock<std::mutex> lock(Queue_Mutex);
        stop = true;
    }
    condition.notify_all();

    for (auto &t: Pool) {
        if (t.joinable())
            t.join();
    }
}

int ThreadManager::getThreadCount() const {
    return Num_Threads.value();
}
