#include "preprocess.h"
#include <opencv2/opencv.hpp>

std::vector<float> preprocess_image(const std::string& path) {

    cv::Mat img = cv::imread(path);
    if (img.empty()) return {};

    cv::resize(img, img, cv::Size(32, 32));

    // BGR → RGB
    cv::cvtColor(img, img, cv::COLOR_BGR2RGB);

    img.convertTo(img, CV_32F, 1.0 / 255.0);

    float mean[3] = {0.4914f, 0.4822f, 0.4465f};
    float std[3]  = {0.2023f, 0.1994f, 0.2010f};

    std::vector<float> tensor(3 * 32 * 32);

    int idx = 0;
    for (int c = 0; c < 3; c++) {
        for (int i = 0; i < 32; i++) {
            for (int j = 0; j < 32; j++) {

                float val = img.at<cv::Vec3f>(i, j)[c];
                val = (val - mean[c]) / std[c];

                tensor[idx++] = val;
            }
        }
    }

    return tensor;
}