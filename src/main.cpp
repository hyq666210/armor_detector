#include <iostream>
#include <opencv2/opencv.hpp>
#include "armor_detector.h"

int main(int argc, char** argv) {
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0]
                  << " <model.onnx> <image.jpg>\n";
        return 1;
    }

    const std::string model_path = argv[1];
    const std::string image_path = argv[2];
    // 判断是视频还是图片
    bool is_video = image_path.size() > 4 &&
        (image_path.substr(image_path.size() - 4) == ".avi" ||
         image_path.substr(image_path.size() - 4) == ".mp4");

    if (is_video) {
        cv::VideoCapture cap(image_path);
        if (!cap.isOpened()) {
            std::cerr << "Cannot open video: " << image_path << "\n";
            return 1;
        }

        int w = (int)cap.get(cv::CAP_PROP_FRAME_WIDTH);
        int h = (int)cap.get(cv::CAP_PROP_FRAME_HEIGHT);
        double fps = cap.get(cv::CAP_PROP_FPS);
        if (fps <= 0) fps = 30.0;
        std::cout << "Video: " << w << "x" << h << " @" << fps << " fps\n";

        cv::VideoWriter writer("output.avi",
            cv::VideoWriter::fourcc('M', 'J', 'P', 'G'),
            fps, cv::Size(w, h));

        // 相机内参
        cv::Mat K = (cv::Mat_<double>(3,3) <<
            2362.2795952460974, 0, 679.17513921636282,
            0, 2336.1916035376717, 529.06858905837635,
            0, 0, 1);
        cv::Mat dist = (cv::Mat_<double>(1,5) <<
            -0.098561592108101559, 0.11173066531932681,
            -0.0038352945867108721, -0.0020421443354436304, 0);

        float armor_w = 0.135f;
        float armor_h = 0.055f;

        ArmorDetector detector(model_path);
        cv::Mat frame;
        int idx = 0;
        while (cap.read(frame)) {
            auto arms = detector.detect(frame);
            for (const auto& a : arms) {
                std::vector<cv::Point> pts;
                for (auto& p : a.corners)
                    pts.emplace_back(cvRound(p.x), cvRound(p.y));
                cv::polylines(frame, pts, true, cv::Scalar(0, 0, 255), 2);

                auto tvec = detector.solvePose(a, K, dist, armor_w, armor_h);
                char buf[64];
                snprintf(buf, sizeof(buf), "%s %.2fm",
                         classNames()[a.cls_id].c_str(), tvec[2]);
                cv::putText(frame, buf,
                    cv::Point(cvRound(a.corners[0].x), cvRound(a.corners[0].y) - 5),
                    cv::FONT_HERSHEY_SIMPLEX, 0.6,
                    cv::Scalar(0, 255, 255), 2);
            }
            writer.write(frame);
            if (++idx % 30 == 0)
                std::cout << "Frame " << idx << "\n";
        }
        cap.release();
        writer.release();
        std::cout << "Saved output.avi (" << idx << " frames)\n";
        return 0;
    }

    cv::Mat img = cv::imread(image_path);
    if (img.empty()) {
        std::cerr << "Failed to load image: " << image_path << "\n";
        return 1;
    }

    try {
        ArmorDetector detector(model_path);
        auto arms = detector.detect(img);
        std::cout << "Detected " << arms.size() << " armor(s)\n";

        // 绘制
                // 相机内参（从群里截图读到的）
        cv::Mat K = (cv::Mat_<double>(3,3) <<
            2362.2795952460974, 0, 679.17513921636282,
            0, 2336.1916035376717, 529.06858905837635,
            0, 0, 1);
        cv::Mat dist = (cv::Mat_<double>(1,5) <<
            -0.098561592108101559, 0.11173066531932681,
            -0.0038352945867108721, -0.0020421443354436304, 0);

        // 装甲板实际尺寸（先用小装甲板 135mm × 55mm）
        float armor_w = 0.135f;
        float armor_h = 0.055f;

        for (const auto& a : arms) {
            // 4 个角点连成四边形
            std::vector<cv::Point> pts;
            for (auto& p : a.corners) pts.emplace_back(cvRound(p.x), cvRound(p.y));
            cv::polylines(img, pts, true, cv::Scalar(0, 0, 255), 2);

            // PnP 解算位置
            auto tvec = detector.solvePose(a, K, dist, armor_w, armor_h);

            // 标注：类别 + 距离（Z 值）
            char buf[64];
            snprintf(buf, sizeof(buf), "%s d=%.2fm", 
                     classNames()[a.cls_id].c_str(), tvec[2]);
            cv::Point2f center = (a.corners[0] + a.corners[2]) * 0.5f;
            cv::putText(img, buf,
                        cv::Point(cvRound(center.x) - 20, cvRound(center.y) - 5),
                        cv::FONT_HERSHEY_SIMPLEX, 0.6,
                        cv::Scalar(0, 255, 255), 2);

            // 打印到终端
            std::cout << "  cls=" << classNames()[a.cls_id]
                      << " conf=" << a.confidence
                      << " pos=(" << tvec[0] << ", " << tvec[1] << ", " << tvec[2] << ") m\n";
        }
        cv::imwrite("result.jpg", img);
        std::cout << "Saved result.jpg\n";
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }
    return 0;
}