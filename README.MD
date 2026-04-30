# QNN Inference Application (C++)

Final Accuracy: ~78% on CIFAR-10 test samples (100 images)

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

7) Softmax-based probability output (correct confidence scores)

8) Top-3 predictions for better interpretability

9) Confusion matrix generation for error analysis

10) CLI-based dataset selection (--dir support)

11) Scalable evaluation on custom datasets

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

qnn_app.exe --backend=opencv --accuracy


# OUTPUT

1) Console logs (timing + predictions)

2) Performance report:

report.txt

# SAMPLE OUTPUT

../../data/images/cat_53.png

Prediction: dog

Confidence: 73.03%

Top-3: dog (0.73) cat (0.24) truck (0.01)


Ground Truth: cat


Accuracy: 78%

===== CONFUSION MATRIX =====

cat -> dog : 4

dog -> cat : 2

bird -> frog : 1

...


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


# PERFORMANCE EVALUATION

Average Preprocessing Time: ~0.40 ms

Average Inference Time: ~0.38 ms

Total Pipeline Time: ~0.78 ms per image


Performance remains stable across larger datasets (100 images)


# KEY OBSERVATIONS

- Strong performance on visually distinct classes (truck, ship, frog)

- Moderate confusion between similar classes (cat vs dog)

- Bird class shows variability due to background noise

- Model generalizes well but limited by architecture complexity

# EXPERIMENTAL SETUP

Dataset: CIFAR-10  

Classes: 10  

Images evaluated: 100  

Platform: Windows + MSVC + CMake  


Model: Custom CNN exported from PyTorch → ONNX → QNN