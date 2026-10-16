#include "traditional_detector.h"
#include <algorithm>
#include <cmath>

TraditionalDetector::TraditionalDetector() {}

// ---------- 颜色分割 ----------
void TraditionalDetector::colorThreshold(const cv::Mat& bgr,
                                         cv::Mat& bin_red,
                                         cv::Mat& bin_blue) const {
    cv::Mat hsv;
    cv::cvtColor(bgr, hsv, cv::COLOR_BGR2HSV);

    // 红色跨 0°，两段合并
    cv::Mat r1, r2;
    cv::inRange(hsv, cv::Scalar(0,   80, 100), cv::Scalar(10,  255, 255), r1);
    cv::inRange(hsv, cv::Scalar(170, 80, 100), cv::Scalar(180, 255, 255), r2);
    bin_red = r1 | r2;

    // 蓝色
    cv::inRange(hsv, cv::Scalar(100, 80, 100), cv::Scalar(130, 255, 255),
                bin_blue);

    // 膨胀
    cv::Mat k = cv::getStructuringElement(cv::MORPH_RECT, {3, 3});
    cv::dilate(bin_red,  bin_red,  k);
    cv::dilate(bin_blue, bin_blue, k);
}

// ---------- 轮廓 → 灯条 ----------
std::vector<cv::RotatedRect>
TraditionalDetector::findLightBars(const cv::Mat& bin,
                                   const cv::Scalar& color) const {
    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(bin, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

    std::vector<cv::RotatedRect> bars;
    for (auto& c : contours) {
        float area = cv::contourArea(c);
        if (area < LIGHT_MIN_AREA || area > LIGHT_MAX_AREA) continue;

        cv::RotatedRect rr = cv::minAreaRect(c);
        float w = rr.size.width;
        float h = rr.size.height;
        float long_side  = std::max(w, h);
        float short_side = std::min(w, h);
        if (short_side < 1e-3f) continue;

        float ratio = long_side / short_side;
        if (ratio < LIGHT_MIN_RATIO || ratio > LIGHT_MAX_RATIO) continue;

        // 填充率
        float fill = area / (w * h);
        if (fill < 0.4f) continue;

        // 统一成"竖灯条"：长边沿 y
        if (w > h) {
            rr.angle += 90.f;
            std::swap(rr.size.width, rr.size.height);
        }
        bars.push_back(rr);
    }
    return bars;
}

// ---------- 灯条配对 ----------
std::vector<Armor>
TraditionalDetector::pairLightBars(const std::vector<cv::RotatedRect>& reds,
                                   const std::vector<cv::RotatedRect>& blues) const {
    // 红蓝灯条不混配。分别配对，再合并
    std::vector<Armor> out;
    std::vector<std::vector<cv::RotatedRect>> groups = { reds, blues };

    for (auto& group : groups) {
        int n = (int)group.size();
        for (int i = 0; i < n; ++i) {
            for (int j = i + 1; j < n; ++j) {
                const auto& a = group[i];
                const auto& b = group[j];

                // 中心距离
                cv::Point2f ca = a.center, cb = b.center;
                float dx = cb.x - ca.x;
                float dy = cb.y - ca.y;
                float gap = std::sqrt(dx*dx + dy*dy);

                // 1) 两灯条中心连线接近水平：|dy| / gap < sin(15°)
                if (std::abs(dy) / gap > std::sin(PAIR_MAX_ANGLE * CV_PI / 180.f))
                    continue;

                // 2) 长度相近
                float la = std::max(a.size.width, a.size.height);
                float lb = std::max(b.size.width, b.size.height);
                float lr = la / lb;
                if (lr < PAIR_LEN_RATIO_MIN || lr > PAIR_LEN_RATIO_MAX) continue;

                // 3) 间距 / 灯条长 在合理范围
                float avg_len = (la + lb) * 0.5f;
                float gr = gap / avg_len;
                if (gr < PAIR_GAP_RATIO_MIN || gr > PAIR_GAP_RATIO_MAX) continue;
                if (gap > 150.f) continue;

                // 4) 高度对齐：两灯条中心 y 差 < 0.5 倍平均灯条长
                if (std::abs(dy) > 0.5f * avg_len) continue;

                // 组装装甲板角点：外接矩形组合
                cv::Point2f pts_a[4], pts_b[4];
                a.points(pts_a);
                b.points(pts_b);

                // 左灯条（x 小的）与右灯条（x 大的）
const cv::RotatedRect& left  = (ca.x < cb.x) ? a : b;
const cv::RotatedRect& right = (ca.x < cb.x) ? b : a;

cv::Point2f lp[4], rp[4];
left.points(lp);
right.points(rp);

// 找每根灯条的最左/最右边界
float xL = std::min({lp[0].x, lp[1].x, lp[2].x, lp[3].x});
float xR = std::max({rp[0].x, rp[1].x, rp[2].x, rp[3].x});

// 高度：取两根灯条 y 范围的中位数并扩展（灯条比装甲板稍长）
float yTop = std::min({lp[0].y, lp[1].y, lp[2].y, lp[3].y,
                       rp[0].y, rp[1].y, rp[2].y, rp[3].y});
float yBot = std::max({lp[0].y, lp[1].y, lp[2].y, lp[3].y,
                       rp[0].y, rp[1].y, rp[2].y, rp[3].y});

cv::Rect2f box(xL, yTop, xR - xL, yBot - yTop);

// 宽高比过滤：装甲板实际宽高比在 1.5~4.0
float box_ratio = box.width / std::max(box.height, 1e-3f);
if (box_ratio < 1.5f || box_ratio > 4.0f) continue;
// 宽度上限：单块装甲板在图像里宽度有限（根据分辨率和距离调）
// tu1 分辨率 960，装甲板约 100~200 像素
if (box.width > 250.f) continue;
 

                Armor arm;
                arm.cls_id = -1;             // 传统方案没有类别（未接数字网络）
                arm.confidence = 1.0f;
                arm.corners = {
                    { box.x,             box.y             },
                    { box.x + box.width, box.y             },
                    { box.x + box.width, box.y + box.height},
                    { box.x,             box.y + box.height}
                };
                arm.box = box;
                out.push_back(arm);
            }
        }
    }
    return out;
}

// ---------- 主入口 ----------
std::vector<Armor> TraditionalDetector::detect(const cv::Mat& bgr) const {
    cv::Mat bin_red, bin_blue;
    colorThreshold(bgr, bin_red, bin_blue);

    auto reds  = findLightBars(bin_red,  cv::Scalar(0, 0, 255));
    auto blues = findLightBars(bin_blue, cv::Scalar(255, 0, 0));

    return pairLightBars(reds, blues);
}