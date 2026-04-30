#pragma once
#include <opencv2/dnn.hpp>
#include <vector>
#include <string>

class OpenCVEngine {
public:
    bool initialize(const std::string& model_path);
    std::vector<float> run_from_image(const std::string& image_path);

private:
    cv::dnn::Net net;
};