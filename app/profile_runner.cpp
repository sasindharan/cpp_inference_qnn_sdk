#include <iostream>
#include <filesystem>
#include <vector>
#include <chrono>

#include "preprocess.h"
#include "qnn_engine.h"
#include "postprocess.h"
#include "profiler.h"

namespace fs = std::filesystem;

int main() {

    std::cout << "[PROFILE MODE]\n";

    Profiler profiler;

    std::string folder = "../../data/images/";
    std::vector<std::string> images;

    for (auto& entry : fs::directory_iterator(folder)) {
        images.push_back(entry.path().string());
    }

    if (images.empty()) {
        std::cout << "No images found\n";
        return -1;
    }

    QNNEngine engine;
    if (!engine.initialize("../../models/compiled/x64/model.dll")) {
        std::cout << "QNN init failed\n";
        return -1;
    }

    for (auto& path : images) {

        // PREPROCESS
        auto t1 = std::chrono::high_resolution_clock::now();
        auto tensor = preprocess_image(path);
        auto t2 = std::chrono::high_resolution_clock::now();

        if (tensor.empty()) continue;

        profiler.add("Preprocess",
            std::chrono::duration<double, std::milli>(t2 - t1).count());

        // INFERENCE
        auto t3 = std::chrono::high_resolution_clock::now();
        auto output = engine.run_inference(tensor);
        auto t4 = std::chrono::high_resolution_clock::now();

        profiler.add("Inference",
            std::chrono::duration<double, std::milli>(t4 - t3).count());

        // POSTPROCESS
        auto t5 = std::chrono::high_resolution_clock::now();
        get_prediction(output);
        auto t6 = std::chrono::high_resolution_clock::now();

        profiler.add("Postprocess",
            std::chrono::duration<double, std::milli>(t6 - t5).count());
    }

    profiler.print();

    return 0;
}