#pragma once
#include <string>
#include <vector>
#include <onnxruntime_cxx_api.h>

// ONNX Runtime 推理封装
class OnnxRunner {
public:
    // 构造时加载模型，读取输入输出名字
    explicit OnnxRunner(const std::string& model_path);

    // 输入 NCHW float 数据，返回第一个输出的原始张量（扁平化）
    std::vector<float> run(const std::vector<float>& input,
                           const std::vector<int64_t>& input_shape);

    // 输入输出维度信息（调试用）
    const std::vector<int64_t>& inputShape() const { return input_shape_; }
    const std::vector<int64_t>& outputShape() const { return output_shape_; }

private:
    Ort::Env env_;
    Ort::Session session_;
    std::string input_name_;
    std::string output_name_;
    std::vector<int64_t> input_shape_;
    std::vector<int64_t> output_shape_;
};