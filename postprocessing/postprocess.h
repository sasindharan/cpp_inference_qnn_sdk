#pragma once
#include <vector>
#include <string>

int get_prediction(const std::vector<float>& output);
void print_prediction(const std::vector<float>& output);
std::string get_class_name(int idx);