#include <iostream>
#include <fstream>

#include <NvOnnxParser.h>

#include "infer.h"
#include "preprocess.h"
#include "postprocess.h"
#include "calibrator.h"
#include "utils.h"
#include <ros/ros.h>
#include <geometry_msgs/Point.h>
#include <InferResult.h>
#include <Results.h>
#include <string>
#include <vector>
#include <sstream>
using namespace nvinfer1;

const char* kInputTensorName = "images";
const char* kOutputTensorName = "output0";
const int kGpuId = 0; //显卡id，一张显卡默认为0

// image的高和宽
const int kInputH = 640;
const int kInputW = 640;
const int kMaxNumOutputBbox = 1000;  // assume the box outputs no more than kMaxNumOutputBbox boxes that conf >= kNmsThresh;

const std::string cacheFile = "./int8.cache";
const std::string calibrationDataPath = "../calibrator";  // 存放用于 int8 量化校准的图像
std::vector<std::string> ClassNames; 

const std::vector<int> track_classes = {0};

const double K[9] = {383.6372985839844, 0.0, 316.88177490234375,
                      0.0, 383.6372985839844, 241.00013732910156,
                      0.0, 0.0, 1.0};
// 模型骨架
const std::vector<std::vector<int>> skeleton {
        {16, 14},
        {14, 12},
        {17, 15},
        {15, 13},
        {12, 13},
        {6, 12},
        {7, 13},
        {6, 7},
        {6, 8},
        {7, 9},
        {8, 10},
        {9, 11},
        {2, 3},
        {1, 2},
        {1, 3},
        {2, 4},
        {3, 5},
        {4, 6},
        {5, 7}
};

YoloDetector::YoloDetector(ros::NodeHandle& nh):
img_(nullptr),
nh_(nh)
{
    nh.getParam("/yolo_node/planFile", trtFile_);
    nh.getParam("/yolo_node/onnxFile", onnxFile_);
    nh.getParam("/yolo_node/confThresh", confThresh_);
    nh.getParam("/yolo_node/numClass", numClass_);
    nh.getParam("/yolo_node/nmsThresh", nmsThresh_);
    nh.getParam("/yolo_node/numKpt", numKpt_);
    nh.getParam("/yolo_node/kptDims", kptDims_);
    nh.getParam("/yolo_node/bFP16Mode", bFP16Mode_);
    nh.getParam("/yolo_node/bINT8Mode", bINT8Mode_);
    nh.getParam("/vClassNames", ClassNames);
    
    numBoxElement_ = 7 + numKpt_ * kptDims_;
    gLogger = Logger(ILogger::Severity::kERROR); // 设置日志记录器
    cudaSetDevice(kGpuId); // 设置当前 GPU

    CHECK(cudaStreamCreate(&stream)); // 创建 CUDA 流

    // 加载 TensorRT 引擎
    get_engine();

    context = engine->createExecutionContext(); // 创建推理上下文
    context->setBindingDimensions(0, Dims32 {4, {1, 3, kInputH, kInputW}}); // 设置输入维度

    // 获取输出维度信息
    Dims32 outDims = context->getBindingDimensions(1);  // 获取输出维度 [1, 84, 8400]
    OUTPUT_CANDIDATES = outDims.d[2];  // 设置输出候选框数量 (8400)
    int outputSize = 1;  // 计算输出数据总大小
    for (int i = 0; i < outDims.nbDims; i++){
        outputSize *= outDims.d[i];
    }

    // 在主机上分配输出数据空间
    outputData = new float[1 + kMaxNumOutputBbox * numBoxElement_];
    // 在设备上分配输入和输出空间
    vBufferD.resize(2, nullptr);
    CHECK(cudaMalloc(&vBufferD[0], 3 * kInputH * kInputW * sizeof(float))); // 输入数据
    CHECK(cudaMalloc(&vBufferD[1], outputSize * sizeof(float))); // 输出数据

    CHECK(cudaMalloc(&transposeDevice, outputSize * sizeof(float))); // 转置数据
    CHECK(cudaMalloc(&decodeDevice, (1 + kMaxNumOutputBbox * numBoxElement_) * sizeof(float))); // 解码数据
}

void YoloDetector::get_engine(){
    if (access(trtFile_.c_str(), F_OK) == 0){ // 检查 TensorRT 文件是否存在
        std::ifstream engineFile(trtFile_, std::ios::binary);
        long int fsize = 0;

        engineFile.seekg(0, engineFile.end);
        fsize = engineFile.tellg();
        engineFile.seekg(0, engineFile.beg);
        std::vector<char> engineString(fsize);
        engineFile.read(engineString.data(), fsize);
        if (engineString.size() == 0) { ROS_INFO("Failed getting serialized engine!"); return;}
        ROS_INFO("Succeeded getting serialized engine!");
        runtime = createInferRuntime(gLogger); // 创建推理运行时
        engine = runtime->deserializeCudaEngine(engineString.data(), fsize); // 反序列化引擎
        if (engine == nullptr) { ROS_INFO("Failed loading engine!"); return; }
        ROS_INFO("Succeeded loading engine!");
    } else {
        IBuilder *            builder     = createInferBuilder(gLogger); // 创建构建器
        INetworkDefinition *  network     = builder->createNetworkV2(1U << int(NetworkDefinitionCreationFlag::kEXPLICIT_BATCH)); // 创建网络定义
        IOptimizationProfile* profile     = builder->createOptimizationProfile(); // 创建优化配置
        IBuilderConfig *      config      = builder->createBuilderConfig(); // 创建构建配置
        config->setMaxWorkspaceSize(1 << 30); // 设置最大工作区大小
        IInt8Calibrator *     pCalibrator = nullptr;
        if (bFP16Mode_){
            config->setFlag(BuilderFlag::kFP16); // 启用 FP16 精度
        }
        if (bINT8Mode_){
            config->setFlag(BuilderFlag::kINT8); // 启用 INT8 精度
            int batchSize = 8;
            pCalibrator = new Int8EntropyCalibrator2(batchSize, kInputW, kInputH, calibrationDataPath.c_str(), cacheFile.c_str()); // 创建 INT8 校准器
            config->setInt8Calibrator(pCalibrator);
        }

        nvonnxparser::IParser* parser = nvonnxparser::createParser(*network, gLogger); // 创建 ONNX 解析器
        if (!parser->parseFromFile(onnxFile_.c_str(), int(gLogger.reportableSeverity))){
           ROS_INFO("Failed parsing .onnx file!");
            for (int i = 0; i < parser->getNbErrors(); ++i){
                auto *error = parser->getError(i);
                std::cout << std::to_string(int(error->code())) << std::string(":") << std::string(error->desc()) << std::endl;
            }
            return;
        }
        ROS_INFO("Succeeded parsing .onnx file!");

        ITensor* inputTensor = network->getInput(0);
        profile->setDimensions(inputTensor->getName(), OptProfileSelector::kMIN, Dims32 {4, {1, 3, kInputH, kInputW}}); // 设置最小尺寸
        profile->setDimensions(inputTensor->getName(), OptProfileSelector::kOPT, Dims32 {4, {1, 3, kInputH, kInputW}}); // 设置最优尺寸
        profile->setDimensions(inputTensor->getName(), OptProfileSelector::kMAX, Dims32 {4, {1, 3, kInputH, kInputW}}); // 设置最大尺寸
        config->addOptimizationProfile(profile); // 添加优化配置文件
        ROS_INFO("Converting .onnx to .plan");
        IHostMemory *engineString = builder->buildSerializedNetwork(*network, *config); // 构建序列化网络
        ROS_INFO("Succeeded building serialized engine!");

        runtime = createInferRuntime(gLogger); // 创建推理运行时
        engine = runtime->deserializeCudaEngine(engineString->data(), engineString->size()); // 反序列化引擎
        if (engine == nullptr) { ROS_INFO("Failed building engine!"); return; }
        ROS_INFO("Succeeded building engine!");

        if (bINT8Mode_ && pCalibrator != nullptr){
            delete pCalibrator; // 删除校准器
        }

        std::ofstream engineFile(trtFile_, std::ios::binary);
        engineFile.write(static_cast<char *>(engineString->data()), engineString->size()); // 保存 .plan 文件
       ROS_INFO("Succeeded saving .plan file!");

        delete engineString; // 释放主机内存
        delete parser; // 释放解析器
        delete config; // 释放构建配置
        delete network; // 释放网络定义
        delete builder; // 释放构建器
    }
}

YoloDetector::~YoloDetector(){
    cudaStreamDestroy(stream); // 销毁 CUDA 流

    for (int i = 0; i < 2; ++i)
    {
        CHECK(cudaFree(vBufferD[i])); // 释放设备内存
    }

    CHECK(cudaFree(transposeDevice)); // 释放设备内存
    CHECK(cudaFree(decodeDevice)); // 释放设备内存

    delete [] outputData; // 释放主机内存

    delete context; // 释放推理上下文
    delete engine; // 释放引擎
    delete runtime; // 释放推理运行时
}

tensorrt_yolo::Results YoloDetector::inference(cv::Mat& img){

    img_ = &img;
    if (img.empty()) return {}; // 如果图像为空，返回空结果

    // 将输入图像数据放到设备上，并进行预处理
    preprocess(img, (float*)vBufferD[0], kInputH, kInputW, stream);

    // 执行 TensorRT 推理
    context->enqueueV2(vBufferD.data(), stream, nullptr);

    // 转置数据 [1 84 8400] 到 [1 8400 84]
    transpose((float*)vBufferD[1], transposeDevice, OUTPUT_CANDIDATES, numClass_ + 4, stream);
    // 解码数据 [1 8400 84] 到 [1 7001]
    decode(transposeDevice, decodeDevice, OUTPUT_CANDIDATES, numClass_, confThresh_, kMaxNumOutputBbox, numBoxElement_, stream);
    // 执行 CUDA 非极大值抑制 (NMS)
    nms(decodeDevice, nmsThresh_, kMaxNumOutputBbox, numBoxElement_, stream);

    // 异步拷贝结果到主机
    CHECK(cudaMemcpyAsync(outputData, decodeDevice, (1 + kMaxNumOutputBbox * numBoxElement_) * sizeof(float), cudaMemcpyDeviceToHost, stream));
    cudaStreamSynchronize(stream); // 等待 CUDA 流完成所有操作

    // 解析检测结果
    tensorrt_yolo::Results vDetections;
    int count = std::min((int)outputData[0], kMaxNumOutputBbox); // 获取检测框数量
    for (int i = 0; i < count; i++){
        int pos = 1 + i * numBoxElement_;
        int keepFlag = (int)outputData[pos + 6];
        if (keepFlag == 1){
            tensorrt_yolo::InferResult det;
            memcpy(det.bbox.data(), &outputData[pos], 4 * sizeof(float)); // 复制边界框数据
            det.conf = outputData[pos + 4]; // 复制置信度
            det.classId = (int)outputData[pos + 5]; // 复制类别 ID
            vDetections.results.push_back(det); // 将检测结果添加到列表中
        }
    }
    // 对检测框进行缩放
    for (size_t j = 0; j < vDetections.results.size(); j++){
        scale_bbox(img, vDetections.results[j].bbox.data());
    }

    return vDetections; // 返回检测结果
}

tensorrt_yolo::Results YoloDetector::inference(cv::Mat& img, bool pose){
    if (img.empty()) return {};
    int nk = numKpt_ * kptDims_;  // number of keypoints total, default 51
    // put input on device, then letterbox、bgr to rgb、hwc to chw、normalize.
    preprocess(img, (float*)vBufferD[0], kInputH, kInputW, stream);

    // tensorrt inference
    context->enqueueV2(vBufferD.data(), stream, nullptr);
    // transpose [56 8400] convert to [8400 56]
    transpose((float*)vBufferD[1], transposeDevice, OUTPUT_CANDIDATES, 4 + numClass_ + nk, stream);
    // convert [8400 56] to [58001, ], 58001 = 1 + 1000 * (4bbox + cond + cls + keepflag + 51kpts)

    decode(transposeDevice, decodeDevice, OUTPUT_CANDIDATES, numClass_, nk, confThresh_, kMaxNumOutputBbox, numBoxElement_, stream);
    // cuda nms
    nms(decodeDevice, nmsThresh_, kMaxNumOutputBbox, numBoxElement_, stream);

    CHECK(cudaMemcpyAsync(outputData, decodeDevice, (1 + kMaxNumOutputBbox * numBoxElement_) * sizeof(float), cudaMemcpyDeviceToHost, stream));
    cudaStreamSynchronize(stream);
    tensorrt_yolo::Results vDetections;
    int count = std::min((int)outputData[0], kMaxNumOutputBbox);

    for (int i = 0; i < count; i++){
        int pos = 1 + i * numBoxElement_;
        int keepFlag = (int)outputData[pos + 6];
        if (keepFlag == 1){
            tensorrt_yolo::InferResult det;
            memcpy(det.bbox.data(), &outputData[pos], 4 * sizeof(float));
            det.conf = outputData[pos + 4];
            det.classId = (int)outputData[pos + 5];
            det.kpts.resize(nk);
            memcpy(det.kpts.data(), &outputData[pos + 7], nk * sizeof(float));
            vDetections.results.push_back(det);
        }
    }

    for (size_t j = 0; j < vDetections.results.size(); j++){
        scale_bbox(img, vDetections.results[j].bbox.data());
        vDetections.results[j].vKpts = scale_kpt_coords(img, vDetections.results[j].kpts.data(), numKpt_, kptDims_);
    }

    return vDetections;
}

