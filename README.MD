# QNN Inference Application (C++)

## OVERVIEW

This project implements a production-style inference pipeline using the Qualcomm Neural Network (QNN) SDK. It performs image classification using a model converted from ONNX and executed via a compiled QNN runtime.

## ARCHITECTURE

Pipeline:
Image → Preprocess → QNN Engine → Inference → Postprocess → Accuracy → Report

## FEATURES

1) CMake-based build system

2) OpenCV preprocessing

3) QNN model integration (DLL)

4) Batch processing of multiple images

5) Runtime flag for accuracy evaluation

6) Performance report generation

## FOLDER STRUCTURE
``` 
    cpp_inference_engine/ 
    │ 
    ├── app/ 
    ├── preprocessing/ 
    ├── postprocessing/ 
    ├── qnn/ 
    ├── models/ 
    ├── data/ 
    ├── docs/ 
    │ └── architecture.png 
    ├── build/ 
    ├── report.txt 
    └── CMakeLists.txt

``` 

# Dataset
Download CIFAR-10 from:

https://www.cs.toronto.edu/~kriz/cifar.html

# BUILD AND RUN INSTRUCTIONS

## 1. Open Developer Command Prompt

## 2. Navigate to project

cd cpp_inference_engine

## 3. Create build directory

mkdir build 

cd build

## 4. Configure

cmake .. ^ 

-DCMAKE_TOOLCHAIN_FILE=../vcpkg/scripts/buildsystems/vcpkg.cmake ^ 

-DOpenCV_DIR=../vcpkg/installed/x64-windows/share/opencv4 ^ 

-G "Visual Studio 18 2026" -A x64

## 5. Build

cmake --build . --config Release

## 6. Run Instructions

cd build\Release 

qnn_app.exe --accuracy


# OUTPUT

1) Console logs (timing + predictions)

2) Performance report:

report.txt


# Accuracy Evaluation

Enable using:

qnn_app.exe --backend=opencv --accuracy

Number of Images Tested: 4  
Accuracy Achieved: 75%

Number of Images Tested: 100  
Accuracy Achieved: 78%

This evaluation provides a more reliable estimate of model performance compared 
to the smaller test set.

Observations:

- Improved stability of accuracy

- Model generalizes reasonably well on unseen data


# Notes

1) Current inference is a placeholder for pipeline validation

2) QNN compiled model is successfully integrated


# Future Improvements

GPU/DSP backend support