#pragma once
#include <opencv2/opencv.hpp>
#include <string>
#include <vector>

// 单个装甲板检测结果
struct Armor {
    int cls_id = -1;                       // 类别索引（0~8）
    float confidence = 0.f;                // 置信度
    std::vector<cv::Point2f> corners;      // 4 个角点，输入尺度像素坐标
    cv::Rect2f box;                        // 包围框
};

// 数字类别名（和 label.txt 一致）
inline const std::vector<std::string>& classNames() {
    static const std::vector<std::string> names = {
        "1", "2", "3", "4", "5", "outpost", "guard", "base", "negative"
    };
    return names;
}