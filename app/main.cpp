#include <iostream>
#include <vector>
#include <string>
#include <filesystem>
#include <chrono>

#include "preprocess.h"
#include "postprocess.h"
#include "qnn_engine.h"
#include "opencv_engine.h"

namespace fs = std::filesystem;

int main(int argc, char* argv[]) {

    std::cout << "[INFO] App Started\n";

    bool accuracy_mode = false;
    std::string backend = "qnn";

    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "--accuracy") accuracy_mode = true;
        if (arg == "--backend=opencv") backend = "opencv";
    }

    std::string folder = "../../data/images/";
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

    for (auto& path : images) {

        auto tensor = preprocess_image(path);
        if (tensor.empty()) continue;

        std::vector<float> output;

        if (backend == "qnn")
            output = qnn_engine.run_inference(tensor);
        else
            output = cv_engine.run_from_image(path);

        int pred = get_predicted_class(output);
        std::string pred_label = classes[pred];

        std::cout << "\n" << path << "\n";
        std::cout << "Prediction: " << pred_label << "\n";

        if (accuracy_mode) {
            std::string gt = get_ground_truth(path, classes);
            std::cout << "Ground Truth: " << gt << "\n";

            if (gt == pred_label) correct++;
        }
    }

    if (accuracy_mode) {
        double acc = (double)correct / images.size() * 100.0;
        std::cout << "\nAccuracy: " << acc << "%\n";
    }

    return 0;
}