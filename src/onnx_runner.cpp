#include "onnx_runner.h"
#include <stdexcept>
#include <iostream>

OnnxRunner::OnnxRunner(const std::string& model_path)
    : env_(ORT_LOGGING_LEVEL_WARNING, "armor"),
      session_(nullptr) {

    Ort::SessionOptions opts;
    opts.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_BASIC);
    session_ = Ort::Session(env_, model_path.c_str(), opts);

    // 读取输入信息
    Ort::AllocatorWithDefaultOptions alloc;
    auto in_name = session_.GetInputNameAllocated(0, alloc);
    input_name_ = in_name.get();
    auto in_info = session_.GetInputTypeInfo(0).GetTensorTypeAndShapeInfo();
    input_shape_ = in_info.GetShape();

    // 读取输出信息
    auto out_name = session_.GetOutputNameAllocated(0, alloc);
    output_name_ = out_name.get();
    auto out_info = session_.GetOutputTypeInfo(0).GetTensorTypeAndShapeInfo();
    output_shape_ = out_info.GetShape();

    std::cout << "[OnnxRunner] model loaded: " << model_path << "\n"
              << "  input : " << input_name_ << " shape=[";
    for (auto d : input_shape_) std::cout << d << ",";
    std::cout << "]\n  output: " << output_name_ << " shape=[";
    for (auto d : output_shape_) std::cout << d << ",";
    std::cout << "]" << std::endl;
}

std::vector<float> OnnxRunner::run(const std::vector<float>& input,
                                   const std::vector<int64_t>& input_shape) {
    Ort::MemoryInfo mem = Ort::MemoryInfo::CreateCpu(
        OrtArenaAllocator, OrtMemTypeDefault);

    // 外部没传 shape 时，用模型里保存的；仍为空则用 [1,3,480,640] 兜底
    std::vector<int64_t> shape = input_shape;
    if (shape.empty()) shape = input_shape_;
    if (shape.empty()) shape = {1, 3, 480, 640};

    Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
        mem,
        const_cast<float*>(input.data()), input.size(),
        shape.data(), shape.size());

    const char* in_names[]  = { input_name_.c_str() };
    const char* out_names[] = { output_name_.c_str() };

    auto outputs = session_.Run(Ort::RunOptions{nullptr},
                                in_names, &input_tensor, 1,
                                out_names, 1);

    auto& out = outputs[0];
    auto info = out.GetTensorTypeAndShapeInfo();
    auto shape_out = info.GetShape();
    size_t n = 1;
    for (auto d : shape_out) n *= d;
    const float* data = out.GetTensorData<float>();
    return std::vector<float>(data, data + n);
}