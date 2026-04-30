#include "postprocess.h"
#include <algorithm>
#include <iostream>
#include <map>

static std::vector<std::string> classes = {
    "airplane","automobile","bird","cat","deer",
    "dog","frog","horse","ship","truck"
};

int get_prediction(const std::vector<float>& output) {
    return std::max_element(output.begin(), output.end()) - output.begin();
}

void print_top3(const std::vector<float>& output) {

    std::vector<std::pair<float,int>> probs;

    for (int i = 0; i < output.size(); i++) {
        probs.push_back({output[i], i});
    }

    std::sort(probs.begin(), probs.end(), std::greater<>());

    std::cout << "Top-3: ";
    for (int i = 0; i < 3; i++) {
        std::cout << classes[probs[i].second]
                  << " (" << probs[i].first << ") ";
    }
    std::cout << std::endl;
}

void print_prediction(const std::vector<float>& output) {

    int pred = get_prediction(output);

    std::cout << "Prediction: " << classes[pred] << std::endl;

    float confidence = output[pred];
    std::cout << "Confidence: " << confidence * 100 << "%" << std::endl;

    print_top3(output);
}

std::string get_class_name(int idx) {
    return classes[idx];
}