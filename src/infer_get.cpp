#include "ros/ros.h"
#include "InferResult.h"
#include <Results.h> //包含自定义消息头文件


//接收并处理消息
void resultsCallback(const tensorrt_yolo::Results::ConstPtr& msg) {
   for (const auto& result : msg->results) {
        ROS_INFO("Confidence: %f", result.conf);
        ROS_INFO("Class ID: %d", result.classId);
        ROS_INFO("Coordinate: [%f, %f, %f]", result.coordinate[0], result.coordinate[1], result.coordinate[2]);
        ROS_INFO("Object ID: %d", result.Id);}
}

int main(int argc, char *argv[]) {
    // 初始化 ROS 节点
    ros::init(argc, argv, "inferresults_sub");
    ros::NodeHandle nh;
    // 订阅话题 /infer_results
    ros::Subscriber results_sub = nh.subscribe<tensorrt_yolo::Results>("infer_results", 10, resultsCallback);
    ros::spin();
    return 0;
}

