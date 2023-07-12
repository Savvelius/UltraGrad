#pragma once

#include <chrono>
#include <iostream>
#include <string>

class Timer {
    typedef std::chrono::high_resolution_clock clock;
    std::string id;
    std::chrono::time_point<clock> start;
public:
    Timer(std::string id_ = "default")
            : start{clock::now()} { id = id_; }
    inline void reset(){
        start = clock::now();
    }
    inline long long get_time(){
        return std::chrono::duration_cast<std::chrono::microseconds>(clock::now() - start).count();
    }
    inline void trigger() {
        auto end = std::chrono::duration_cast<std::chrono::microseconds>(clock::now() - start).count();
        std::cout << "Timer(" << id << ") triggered. Duration = "<< end
                  << " [10^-6 seconds]." << std::endl;
    }

    ~Timer() {
        auto end = std::chrono::duration_cast<std::chrono::microseconds>(clock::now() - start).count();
        std::cout << "Timer(" << id << ") died. Duration = "<< end
                  << " [10^-6 seconds]." << std::endl;
    }
};