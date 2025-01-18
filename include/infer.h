#ifndef INFER_H
#define INFER_H

#include <opencv2/opencv.hpp>
#include "public.h"
#include "types.h"
#include <ros/ros.h>
#include <InferResult.h>
#include <Results.h>

using namespace nvinfer1;

extern const char* kInputTensorName;
extern const char* kOutputTensorName;
extern const int kGpuId;
extern const int kInputH ;
extern const int kInputW ;
extern const int kMaxNumOutputBbox;  // assume the box outputs no more than kMaxNumOutputBbox boxes that conf >= kNmsThresh;

extern const std::string cacheFile;
extern const std::string calibrationDataPath ;  // 存放用于 int8 量化校准的图像
extern std::vector<std::string> ClassNames; 

extern const std::vector<int> track_classes;
extern const double K[9];
// 模型骨架
extern const std::vector<std::vector<int>> skeleton;

// std::vector<double> D = {0.0, 0.0, 0.0, 0.0};

class YoloDetector
{
public:
    YoloDetector(ros::NodeHandle& nh);
    ~YoloDetector();
    tensorrt_yolo::Results inference(cv::Mat& img);
    tensorrt_yolo::Results inference(cv::Mat& img, bool pose);
    ros::NodeHandle&    nh_;
private:
    void get_engine();
    void deserialize_engine();
    void serialize_engine();

private:

    Logger              gLogger;
    std::string         trtFile_;
    std::string         onnxFile_;
    std::string         vClassNamesStr;
    bool                bFP16Mode_;
    bool                bINT8Mode_;

    int                 numClass_;
    float               nmsThresh_;
    float               confThresh_;
    int                 numKpt_;
    int                 kptDims_;
    int                 numBoxElement_;
    ICudaEngine *       engine;
    IRuntime *          runtime;
    IExecutionContext * context;

    cudaStream_t        stream;

    float *             outputData;
    std::vector<void *> vBufferD;
    float *             transposeDevice;
    float *             decodeDevice;

    int                 OUTPUT_CANDIDATES;  // 8400: 80 * 80 + 40 * 40 + 20 * 20
    cv::Mat*            img_;

};

#endif  // INFER_H
