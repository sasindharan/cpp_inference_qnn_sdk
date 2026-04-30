#pragma once
#include <map>
#include <string>
#include <iostream>

class Profiler {
public:
    void add(const std::string& name, double ms) {
        data[name] += ms;
        count[name]++;
    }

    void print() {
        std::cout << "\n===== LAYER-WISE PROFILING =====\n";
        for (auto& kv : data) {
            double avg = kv.second / count[kv.first];
            std::cout << kv.first << ": " << avg << " ms\n";
        }
    }

private:
    std::map<std::string, double> data;
    std::map<std::string, int> count;
};