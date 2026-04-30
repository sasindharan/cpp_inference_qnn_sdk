#include <iostream>
#include <vector>
#include <string>
#include <filesystem>
#include <map>
#include <cmath>
#include <algorithm>

#include "preprocess.h"
#include "postprocess.h"
#include "qnn_engine.h"
#include "opencv_engine.h"

namespace fs = std::filesystem;

// ✅ FULL CIFAR-10 ground truth mapping
std::string get_ground_truth(const std::string& path) {

    if (path.find("airplane") != std::string::npos) return "airplane";
    if (path.find("automobile") != std::string::npos) return "automobile";
    if (path.find("bird") != std::string::npos) return "bird";
    if (path.find("cat") != std::string::npos) return "cat";
    if (path.find("deer") != std::string::npos) return "deer";
    if (path.find("dog") != std::string::npos) return "dog";
    if (path.find("frog") != std::string::npos) return "frog";
    if (path.find("horse") != std::string::npos) return "horse";
    if (path.find("ship") != std::string::npos) return "ship";
    if (path.find("truck") != std::string::npos) return "truck";

    return "unknown";
}

int main(int argc, char* argv[]) {

    std::cout << "[INFO] App Started\n";

    bool accuracy_mode = false;
    std::string backend = "qnn";
    std::string folder = "../../data/images";

    // ✅ CLI parsing
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];

        if (arg == "--accuracy") accuracy_mode = true;
        if (arg == "--backend=opencv") backend = "opencv";

        if (arg.find("--dir=") != std::string::npos) {
            folder = arg.substr(6);
        }
    }

    std::vector<std::string> images;
    for (auto& entry : fs::directory_iterator(folder)) {
        images.push_back(entry.path().string());
    }

    std::vector<std::string> classes = {
        "airplane","automobile","bird","cat","deer",
        "dog","frog","horse","ship","truck"
    };

    QNNEngine qnn_engine;
    OpenCVEngine cv_engine;

    if (backend == "qnn") {
        std::cout << "[BACKEND] QNN\n";
        qnn_engine.initialize("../../models/compiled/x64/model.dll");
    } else {
        std::cout << "[BACKEND] OpenCV\n";
        if (!cv_engine.initialize("../../models/model.onnx")) return -1;
    }

    int correct = 0;
    std::map<std::string, std::map<std::string,int>> confusion;

    for (auto& path : images) {

        std::vector<float> output;

        if (backend == "qnn") {
            auto tensor = preprocess_image(path);
            if (tensor.empty()) continue;
            output = qnn_engine.run_inference(tensor);
        } else {
            output = cv_engine.run_from_image(path);
        }

        // 🔥 SOFTMAX (FIXED)
        std::vector<float> probs(output.size());
        float sum = 0.0f;

        for (float v : output) {
            sum += std::exp(v);
        }

        for (int i = 0; i < output.size(); i++) {
            probs[i] = std::exp(output[i]) / sum;
        }

        int pred = std::max_element(probs.begin(), probs.end()) - probs.begin();
        std::string pred_label = classes[pred];

        std::cout << "\n" << path << "\n";

        // ✅ Confidence
        float confidence = probs[pred];
        std::cout << "Prediction: " << pred_label << "\n";
        std::cout << "Confidence: " << confidence * 100 << "%\n";

        // ✅ Top-3
        std::vector<std::pair<float,int>> top_probs;

        for (int i = 0; i < probs.size(); i++) {
            top_probs.push_back({probs[i], i});
        }

        std::sort(top_probs.begin(), top_probs.end(), std::greater<>());

        std::cout << "Top-3: ";
        for (int i = 0; i < 3; i++) {
            std::cout << classes[top_probs[i].second]
                      << " (" << top_probs[i].first << ") ";
        }
        std::cout << "\n";

        // ✅ Accuracy + Confusion
        if (accuracy_mode) {
            std::string gt = get_ground_truth(path);
            std::cout << "Ground Truth: " << gt << "\n";

            confusion[gt][pred_label]++;

            if (gt == pred_label) correct++;
        }
    }

    if (accuracy_mode) {
        double acc = (double)correct / images.size() * 100.0;
        std::cout << "\nAccuracy: " << acc << "%\n";

        std::cout << "\n===== CONFUSION MATRIX =====\n";
        for (auto& gt : confusion) {
            for (auto& pred : gt.second) {
                std::cout << gt.first << " -> " << pred.first
                          << " : " << pred.second << "\n";
            }
        }
    }

    return 0;
}