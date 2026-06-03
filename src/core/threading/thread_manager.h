#ifndef BUMPERDETECTION_THREADMANAGER_H
#define BUMPERDETECTION_THREADMANAGER_H
#include <thread>
#include <vector>
#include <mutex>
#include <queue>
#include <condition_variable>
#include <functional>
#include <optional>
#include <future>
#include <algorithm>

class ThreadManager {
public:
    explicit ThreadManager(int thread_count);
    ~ThreadManager();

    template<typename F>
    auto enqueue(F &&func) -> std::future<decltype(func())> {
        using ReturnType = decltype(func());

        auto promise = std::make_shared<std::promise<ReturnType> >();
        auto future = promise->get_future();

        addJob([func = std::forward<F>(func), p = std::move(promise)]() mutable {
            try {
                if constexpr (std::is_void_v<ReturnType>) {
                    func();
                    p->set_value();
                } else {
                    p->set_value(func());
                }
            } catch (...) {
                p->set_exception(std::current_exception());
            }
        });

        return future;
    }

    void workerLoop();
    void addJob(const std::function<void()> &New_Job);
    void shutdown();

    int getThreadCount() const;

    ThreadManager(const ThreadManager &) = delete;

    ThreadManager &operator=(const ThreadManager &) = delete;

    ThreadManager(ThreadManager &&) = delete;

    ThreadManager &operator=(ThreadManager &&) = delete;

private:
    std::optional<int> Num_Threads;

    std::vector<std::thread> Pool;


    std::mutex Queue_Mutex;
    std::queue<std::function<void()> > Queue;
    std::condition_variable condition;

    std::atomic<bool> stop = false;
};


#endif
