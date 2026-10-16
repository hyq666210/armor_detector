#pragma once
#include <opencv2/opencv.hpp>
#include <vector>
#include "onnx_runner.h"
#include "types.h"

// 装甲板网络解码器
class ArmorDetector {
public:
    // 输入尺寸（和网络一致）
    static constexpr int IN_W = 640;
    static constexpr int IN_H = 480;

    // 输出解码参数
    static constexpr float CONF_THRESH = 0.5f;   // 类别置信度阈值
    static constexpr float NMS_IOU     = 0.45f;  // NMS IoU 阈值
    static constexpr int   NUM_CLS     = 9;      // 类别数
    static constexpr int   NUM_KPT     = 4;      // 关键点数

    explicit ArmorDetector(const std::string& model_path);

    // 检测：输入原图，输出装甲板列表（角点已映射回原图坐标）
    std::vector<Armor> detect(const cv::Mat& bgr);
    cv::Vec3d solvePose(const Armor& arm,
                        const cv::Mat& K,
                        const cv::Mat& dist,
                        float armor_w,
                        float armor_h) const;

private:
    // 预处理：BGR → resize → RGB → 归一化 → NCHW
    std::vector<float> preprocess(const cv::Mat& bgr) const;

    // 解码：[1,21,6300] → 候选 Armor（坐标在 640×480 系）
    std::vector<Armor> decode(const std::vector<float>& raw) const;

    // 应用 NMS
    std::vector<Armor> nms(const std::vector<Armor>& arms) const;

    // 坐标从 640×480 映射回原图
    void remapToOriginal(std::vector<Armor>& arms,
                         const cv::Size& orig) const;

    OnnxRunner runner_;
};