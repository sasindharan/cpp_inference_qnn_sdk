#pragma once
#include <vector>
#include <string>

int get_predicted_class(const std::vector<float>& output);

std::string get_ground_truth(const std::string& path,
                            const std::vector<std::string>& classes);