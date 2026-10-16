# armor_detector

RoboMaster 视觉第二周作业：基于神经网络的装甲板识别与 PnP 位姿解算。

## 依赖
- OpenCV 4.10（`sudo apt install libopencv-dev`）
- ONNX Runtime 1.31（下载到 `/opt/onnxruntime`）
- CMake 3.16+，GCC 15+

## 构建
```bash
mkdir build && cd build
cmake ..
make -j$(nproc)
```

## 运行
```bash
# 图片
./armor_detector <model.onnx> <image.jpg>

# 视频
./armor_detector <model.onnx> <input.avi>
```

## 模型
从 [SZURPVision/26_NNDeployment_Lib_and_Detection_Models](https://github.com/SZURPVision/26_NNDeployment_Lib_and_Detection_Models) 下载
`所有模型/onnx/Infantry-v8n-fp16-20260726-D1.8w-B16.onnx`。

## 项目结构
```
├── CMakeLists.txt
├── include/
│   ├── armor_detector.h    # 检测器接口
│   ├── onnx_runner.h       # ONNX Runtime 封装
│   └── types.h             # 数据结构
└── src/
    ├── armor_detector.cpp  # 解码 + NMS + PnP
    ├── main.cpp            # 入口
    └── onnx_runner.cpp     # 推理实现
```