#ifndef BUMPERDETECTION_THREADMANAGER_H
#define BUMPERDETECTION_THREADMANAGER_H
#include <thread>
#include <vector>
#include <mutex>
#include <queue>
#include <condition_variable>
#include <functional>
#include <tesseract/baseapi.h>
#include <optional>
#include <leptonica/allheaders.h>
#include <future>
#include <algorithm>

#include "debug_log.h"


class ThreadManager
{
public:
    explicit ThreadManager(int thread_count)
    {
        if (!Num_Threads.has_value()) Num_Threads = thread_count;
        if (thread_count > std::thread::hardware_concurrency()) {
            Num_Threads = std::thread::hardware_concurrency();
            logger->critical("Too many threads specified, defaulting to 1...");
        }
        if (thread_count < 1) {
            Num_Threads = 1;
        }

        for (int i = 0; i < Num_Threads; i++)
        {
            Pool.emplace_back(&ThreadManager::workerLoop, this);
        }
    }

    ~ThreadManager() {
        shutdown();
    }

    void workerLoop()
    {
        while (true)
        {
            std::function<void()> Job;
            {
                std::unique_lock<std::mutex> lock(Queue_Mutex);

                condition.wait(lock, [this]
                               { return stop || !Queue.empty(); });
                if (stop && Queue.empty())
                    return;

                Job = Queue.front();
                Queue.pop();
            }
            Job();
        }
    }

    template <typename F>
    auto enqueue(F &&func) -> std::future<decltype(func())>
    {
        using ReturnType = decltype(func());

        auto promise = std::make_shared<std::promise<ReturnType>>();
        auto future = promise->get_future();

        addJob([func = std::forward<F>(func), p = std::move(promise)]() mutable
               {
            try{
                if constexpr (std::is_void_v<ReturnType>){
                    func();
                    p->set_value();
                }
                else {
                    p->set_value(func());
                }
            }
            catch (...){
                p->set_exception(std::current_exception());
            } });

        return future;
    }

    void addJob(const std::function<void()>& New_Job)
    {
        {
            std::unique_lock<std::mutex> lock(Queue_Mutex);
            if (stop)
                return;

            Queue.push(New_Job);
        }
        condition.notify_one();
    }

    void shutdown()
    {
        if (stop) return;
        {
            std::unique_lock<std::mutex> lock(Queue_Mutex);
            stop = true;
        }
        condition.notify_all();

        for (auto &t : Pool)
        {
            if (t.joinable())
                t.join();
        }
    }

    std::optional<int> Num_Threads;
private:
    std::vector<std::thread> Pool;


    std::mutex Queue_Mutex;
    std::queue<std::function<void()>> Queue;
    std::condition_variable condition;

    std::atomic<bool> stop = false;
};

#endif