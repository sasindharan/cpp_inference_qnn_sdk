#pragma once
#include <vector>
#include <string>
#include <windows.h>

class QNNEngine {
public:
    bool initialize(const std::string& model_path);
    std::vector<float> run_inference(const std::vector<float>& input);

private:
    HMODULE dll_handle = nullptr;
};