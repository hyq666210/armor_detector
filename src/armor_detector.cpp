#include "armor_detector.h"
#include <algorithm>
#include <cmath>
#include <iostream>

// ---------- 构造 ----------
ArmorDetector::ArmorDetector(const std::string& model_path)
    : runner_(model_path) {}

// ---------- 预处理 ----------
std::vector<float> ArmorDetector::preprocess(const cv::Mat& bgr) const {
    cv::Mat resized, rgb;
    cv::resize(bgr, resized, cv::Size(IN_W, IN_H));
    cv::cvtColor(resized, rgb, cv::COLOR_BGR2RGB);

    // HWC uint8 → CHW float [0,1]
    std::vector<float> blob(3 * IN_H * IN_W);
    const int H = IN_H, W = IN_W;
    for (int y = 0; y < H; ++y) {
        const uchar* row = rgb.ptr<uchar>(y);
        for (int x = 0; x < W; ++x) {
            blob[0 * H * W + y * W + x] = row[x * 3 + 0] / 255.f;  // R
            blob[1 * H * W + y * W + x] = row[x * 3 + 1] / 255.f;  // G
            blob[2 * H * W + y * W + x] = row[x * 3 + 2] / 255.f;  // B
        }
    }
    return blob;
}

// ---------- 解码 ----------
std::vector<Armor> ArmorDetector::decode(const std::vector<float>& raw) const {
    // raw 布局：[1, 21, 6300]，展平为一维。索引方式：raw[c * 6300 + n]
    std::cout << "[decode] raw size = " << raw.size()
              << " (expect " << 21 * 6300 << ")\n";
    constexpr int C = 21;
    constexpr int N = 6300;
    std::vector<Armor> result;

    for (int n = 0; n < N; ++n) {
        // 找 9 类里最大分数
        int best_cls = -1;
        float best_score = 0.f;
        for (int c = 0; c < NUM_CLS; ++c) {
        float s = raw[(4 + c) * N + n]; 
        if (s > best_score) { best_score = s; best_cls = c; }  // 跳过 4 个框参数            if (s > best_score) { best_score = s; best_cls = c; }
        }
        // 过滤：negative 类（索引 8）不要；低置信度不要
        if (best_cls < 0 || best_cls == 8) continue;
        if (best_score < CONF_THRESH) continue;

        // 取 4 个角点
        Armor arm;
        arm.cls_id = best_cls;
        arm.confidence = best_score;
        for (int k = 0; k < NUM_KPT; ++k) {
            float px = raw[(13 + k * 2 + 0) * N + n];
            float py = raw[(13 + k * 2 + 1) * N + n];
            arm.corners.emplace_back(px, py);
        }
        // 计算包围框
        arm.box = cv::boundingRect(arm.corners);
        result.push_back(std::move(arm));
    }
    return result;
}

// ---------- NMS ----------
std::vector<Armor> ArmorDetector::nms(const std::vector<Armor>& arms) const {
    std::vector<int> idx(arms.size());
    for (size_t i = 0; i < idx.size(); ++i) idx[i] = i;
    std::sort(idx.begin(), idx.end(), [&](int a, int b) {
        return arms[a].confidence > arms[b].confidence;
    });

    std::vector<bool> removed(arms.size(), false);
    std::vector<Armor> keep;
    for (size_t i = 0; i < idx.size(); ++i) {
        if (removed[idx[i]]) continue;
        keep.push_back(arms[idx[i]]);
        for (size_t j = i + 1; j < idx.size(); ++j) {
            if (removed[idx[j]]) continue;
            // 同类别才比较（不同板可以重叠）
            if (arms[idx[i]].cls_id != arms[idx[j]].cls_id) continue;
            float inter = (arms[idx[i]].box & arms[idx[j]].box).area();
            float uni   = (arms[idx[i]].box | arms[idx[j]].box).area();
            float iou   = inter / std::max(uni, 1e-6f);
            if (iou > NMS_IOU) removed[idx[j]] = true;
        }
    }
    return keep;
}

// ---------- 坐标映射回原图 ----------
void ArmorDetector::remapToOriginal(std::vector<Armor>& arms,
                                    const cv::Size& orig) const {
    float sx = static_cast<float>(orig.width)  / IN_W;
    float sy = static_cast<float>(orig.height) / IN_H;
    for (auto& a : arms) {
        for (auto& p : a.corners) { p.x *= sx; p.y *= sy; }
        a.box = cv::boundingRect(a.corners);
    }
}

// ---------- 主入口 ----------
std::vector<Armor> ArmorDetector::detect(const cv::Mat& bgr) {
    auto blob = preprocess(bgr);
    std::vector<int64_t> shape = {1, 3, IN_H, IN_W};
    auto raw = runner_.run(blob, shape);
    auto arms = decode(raw);
    arms = nms(arms);
    remapToOriginal(arms, bgr.size());
    return arms;
}


// ---------- PnP 位姿解算 ----------
cv::Vec3d ArmorDetector::solvePose(const Armor& arm,
                                   const cv::Mat& K,
                                   const cv::Mat& dist,
                                   float armor_w,
                                   float armor_h) const {
    // 装甲板 3D 点：以几何中心为原点，Z=0 平面
    float hw = armor_w / 2.f;
    float hh = armor_h / 2.f;

    // 顺序：[左上, 右上, 右下, 左下]（逆时针，相机系 Y 向下）
    std::vector<cv::Point3f> obj = {
        {-hw, -hh, 0},
        { hw, -hh, 0},
        { hw,  hh, 0},
        {-hw,  hh, 0}
    };

    // 2D 点：直接用 4 个角点
    std::vector<cv::Point2f> img(arm.corners.begin(), arm.corners.end());

    if (img.size() != 4) return cv::Vec3d(0, 0, 0);

    cv::Vec3d rvec, tvec;
    bool ok = cv::solvePnP(obj, img, K, dist, rvec, tvec,
                           false, cv::SOLVEPNP_IPPE);
    if (!ok) return cv::Vec3d(0, 0, 0);
    return tvec;
}