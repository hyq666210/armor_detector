cmake_minimum_required(VERSION 3.16)
project(armor_detector CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(OpenCV REQUIRED)

set(ONNXRUNTIME_ROOT "/opt/onnxruntime")
include_directories(${ONNXRUNTIME_ROOT}/include)
link_directories(${ONNXRUNTIME_ROOT}/lib)

add_executable(armor_detector
    src/main.cpp
    src/onnx_runner.cpp
    src/armor_detector.cpp
)

target_include_directories(armor_detector PRIVATE include)
target_link_libraries(armor_detector
    ${OpenCV_LIBS}
    onnxruntime
)
# ===== 传统视觉可执行文件 =====
add_executable(armor_detector_trad
    src/traditional_main.cpp
    src/traditional_detector.cpp
    src/onnx_runner.cpp
)
target_include_directories(armor_detector_trad PRIVATE include)
target_link_libraries(armor_detector_trad
    ${OpenCV_LIBS}
    onnxruntime
)