cmake_minimum_required(VERSION 3.10)
project(QNN_Inference_App)

set(CMAKE_CXX_STANDARD 17)

if(MSVC)
    add_definitions(-D_CRT_SECURE_NO_WARNINGS)
endif()

# Find OpenCV
find_package(OpenCV REQUIRED)

include_directories(
    ${PROJECT_SOURCE_DIR}/utils/include
    ${PROJECT_SOURCE_DIR}/qnn
    ${PROJECT_SOURCE_DIR}/preprocessing
    ${PROJECT_SOURCE_DIR}/postprocessing
    ${OpenCV_INCLUDE_DIRS}
)

# Common source files
file(GLOB COMMON_SOURCES
    ${PROJECT_SOURCE_DIR}/qnn/*.cpp
    ${PROJECT_SOURCE_DIR}/preprocessing/*.cpp
    ${PROJECT_SOURCE_DIR}/postprocessing/*.cpp
    ${PROJECT_SOURCE_DIR}/utils/src/*.cpp
)

# Main app
add_executable(qnn_app
    ${PROJECT_SOURCE_DIR}/app/main.cpp
    ${COMMON_SOURCES}
)

target_link_libraries(qnn_app ${OpenCV_LIBS})

# Profiling app
add_executable(profile_runner
    ${PROJECT_SOURCE_DIR}/app/profile_runner.cpp
    ${COMMON_SOURCES}
)

target_link_libraries(profile_runner ${OpenCV_LIBS})