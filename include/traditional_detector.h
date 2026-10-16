#pragma once
#include <opencv2/opencv.hpp>
#include <vector>
#include <string>
#include "onnx_runner.h"
#include "types.h"

// 传统视觉装甲板检测器
class TraditionalDetector {
public:
    // 灯条检测参数
    static constexpr float LIGHT_MIN_RATIO  = 1.5f;   // 灯条长宽比下限
    static constexpr float LIGHT_MAX_RATIO  = 10.0f;  // 上限
    static constexpr float LIGHT_MIN_AREA   = 20.f;   // 最小面积
    static constexpr float LIGHT_MAX_AREA   = 5000.f; // 最大面积

    // 配对参数
    static constexpr float PAIR_MAX_ANGLE   = 15.f;   // 两灯条最大角度差（度）
    static constexpr float PAIR_LEN_RATIO_MIN = 0.6f; // 长度比下限
    static constexpr float PAIR_LEN_RATIO_MAX = 1.7f; // 上限
    static constexpr float PAIR_GAP_RATIO_MIN = 1.0f; // 间距/灯条长 下限
    static constexpr float PAIR_GAP_RATIO_MAX = 5.0f; // 上限

    TraditionalDetector();

    // 检测：输入 BGR 原图，输出装甲板列表
    std::vector<Armor> detect(const cv::Mat& bgr) const;

private:
    // 颜色分割 → 二值图
    void colorThreshold(const cv::Mat& bgr,
                        cv::Mat& bin_red,
                        cv::Mat& bin_blue) const;

    // 轮廓 → 灯条候选
    std::vector<cv::RotatedRect>
    findLightBars(const cv::Mat& bin, const cv::Scalar& color) const;

    // 灯条配对 → 装甲板候选
    std::vector<Armor>
    pairLightBars(const std::vector<cv::RotatedRect>& reds,
                  const std::vector<cv::RotatedRect>& blues) const;
};