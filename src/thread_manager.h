#ifndef BUMPERDETECTION_THREADMANAGER_H
#define BUMPERDETECTION_THREADMANAGER_H
#include <thread>
#include <vector>
#include <mutex>
#include <queue>
#include <condition_variable>
#include <functional>
#include <tesseract/baseapi.h>
#include <leptonica/allheaders.h>
#include <future>
#include <algorithm>

class ThreadManager
{
public:
    ThreadManager()
    {
        for (int i = 0; i < Num_Threads; i++)
        {
            Pool.emplace_back(std::thread(&ThreadManager::workerLoop, this));
        }
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

    void addJob(std::function<void()> New_Job)
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
    
    int numThreads = Num_Threads;
private:
int Num_Threads = std::max(1u, std::thread::hardware_concurrency()); // config

    std::vector<std::thread> Pool;

    std::mutex Queue_Mutex;
    std::queue<std::function<void()>> Queue;
    std::condition_variable condition;

    std::atomic<bool> stop = false;
};

#endif