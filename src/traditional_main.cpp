#include <iostream>
#include <opencv2/opencv.hpp>
#include "traditional_detector.h"

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <image_or_video>\n";
        return 1;
    }
    std::string path = argv[1];

    TraditionalDetector detector;

    // 判断图片 or 视频
    bool is_video = path.size() > 4 &&
        (path.substr(path.size()-4) == ".avi" ||
         path.substr(path.size()-4) == ".mp4");

    if (is_video) {
        cv::VideoCapture cap(path);
        if (!cap.isOpened()) { std::cerr << "Cannot open video\n"; return 1; }

        int w = (int)cap.get(cv::CAP_PROP_FRAME_WIDTH);
        int h = (int)cap.get(cv::CAP_PROP_FRAME_HEIGHT);
        double fps = cap.get(cv::CAP_PROP_FPS);
        if (fps <= 0) fps = 30;

        cv::VideoWriter writer("output_trad.avi",
            cv::VideoWriter::fourcc('M','J','P','G'),
            fps, cv::Size(w, h));

        cv::Mat frame;
        int idx = 0;
        while (cap.read(frame)) {
            auto arms = detector.detect(frame);
            for (const auto& a : arms) {
                std::vector<cv::Point> pts;
                for (auto& p : a.corners)
                    pts.emplace_back(cvRound(p.x), cvRound(p.y));
                cv::polylines(frame, pts, true, cv::Scalar(0, 255, 0), 2);
            }
            writer.write(frame);
            if (++idx % 30 == 0) std::cout << "Frame " << idx << "\n";
        }
        cap.release();
        writer.release();
        std::cout << "Saved output_trad.avi (" << idx << " frames)\n";
        return 0;
    }

    // 图片
    cv::Mat img = cv::imread(path);
    if (img.empty()) { std::cerr << "Cannot open image\n"; return 1; }

    auto arms = detector.detect(img);
    std::cout << "Detected " << arms.size() << " armor(s)\n";
    for (const auto& a : arms) {
        std::vector<cv::Point> pts;
        for (auto& p : a.corners)
            pts.emplace_back(cvRound(p.x), cvRound(p.y));
        cv::polylines(img, pts, true, cv::Scalar(0, 255, 0), 2);
    }
    cv::imwrite("result_trad.jpg", img);
    std::cout << "Saved result_trad.jpg\n";
    return 0;
}