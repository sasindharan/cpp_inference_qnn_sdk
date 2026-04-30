#include "postprocess.h"
#include <algorithm>
#include <filesystem>

int get_predicted_class(const std::vector<float>& output) {
    return std::distance(output.begin(),
                         std::max_element(output.begin(), output.end()));
}

std::string get_ground_truth(const std::string& path,
                            const std::vector<std::string>& classes) {

    std::string filename = std::filesystem::path(path).filename().string();

    for (const auto& cls : classes) {
        if (filename.find(cls) != std::string::npos) {
            return cls;
        }
    }

    return "unknown";
}