#include "opencv_engine.h"
#include <opencv2/opencv.hpp>
#include <iostream>
#include <cstring>

bool OpenCVEngine::initialize(const std::string& model_path) {
    net = cv::dnn::readNetFromONNX(model_path);
    std::cout << "[OpenCV] Model loaded successfully\n";
    return true;
}

std::vector<float> OpenCVEngine::run_from_image(const std::string& path) {

    cv::Mat img = cv::imread(path);
    if (img.empty()) return {};

    // Resize
    cv::resize(img, img, cv::Size(32, 32));

    // BGR → RGB
    cv::cvtColor(img, img, cv::COLOR_BGR2RGB);

    // Scale
    img.convertTo(img, CV_32F, 1.0 / 255.0);

    // Normalize
    float mean[3] = {0.4914f, 0.4822f, 0.4465f};
    float stdv[3] = {0.2023f, 0.1994f, 0.2010f};

    std::vector<float> tensor(3 * 32 * 32);

    int idx = 0;
    for (int c = 0; c < 3; c++) {
        for (int i = 0; i < 32; i++) {
            for (int j = 0; j < 32; j++) {

                float val = img.at<cv::Vec3f>(i, j)[c];
                val = (val - mean[c]) / stdv[c];

                tensor[idx++] = val;
            }
        }
    }

    // Create blob manually (NCHW)
    cv::Mat blob(1, 3 * 32 * 32, CV_32F, tensor.data());
    blob = blob.reshape(1, {1, 3, 32, 32});

    net.setInput(blob);

    cv::Mat output = net.forward();

    std::vector<float> result(output.total());
    std::memcpy(result.data(), output.ptr<float>(),
                output.total() * sizeof(float));

    return result;
}