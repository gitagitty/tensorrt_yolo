# ROS1中通过 TensorRT部署YOLOv8 目标检测模型

## 更新日志

### v3.2 -2025.1.18

- 完成对config功能的迁移

### v4.0 -2024.8.27

- [完成ros2的兼容](https://github.com/wyf-yfw/TensorRT_YOLO_ROS2)
- 分离msg和主函数部分

### v3.1 -2024.8.26

- 完成对TensorRT 10的兼容

### v3.0 - 2024.8.25

- 大量config参数从config.cpp文件转移到launch文件当中,方便参数调整
- 合并多个publish为一
- 增加pose检测

### v2.1 - 2024.8.22

- 合并d_camera_infer_node和camera_infer_node，统一使用camera_infer_node
- 增加depth变量

### v2.0 -  2024.8.19

- 加入bytetrack算法
- 增加d_camera_infer_node和camera_infer_node的track功能

### v1.1 - 2024.8.14

- 删除用于表示检测结果的type.h文件
- 增加infer_result.msg和results.msg文件用于表示和publish检测结果
- 实现检测结果在ros上的publish
- 增加d435i_yolo.launch文件
- 增加track变量
- 完善config文件
- 完善readme

### v1.0 - 2024.8.3

- 实现基本的相机和照片的目标检测功能

## 实现效果

### [目标检测](https://www.bilibili.com/video/BV1Niv9erEuw/?spm_id_from=333.788.recommend_more_video.0&vd_source=bb696fabd15eaa2a7c74687a5ff42a1b)


### [目标追踪](https://www.bilibili.com/video/BV1NmpSeeE2o/?spm_id_from=333.788.recommend_more_video.1&vd_source=bb696fabd15eaa2a7c74687a5ff42a1b)


### [pose检测](https://www.bilibili.com/video/BV1fzWCeBEPx/?vd_source=bb696fabd15eaa2a7c74687a5ff42a1b)



## 环境
#### 支持[TensorRT 8](https://github.com/wyf-yfw/TensorRT_YOLO_ROS/releases/tag/v12.8.1)

#### 支持[TensorRT 10](https://github.com/wyf-yfw/TensorRT_YOLO_ROS/releases/tag/v12.10.1)
#### [2025版](https://github.com/gitagitty/tensorrt_yolo.git)


#### [ROS2仓库](https://github.com/wyf-yfw/TensorRT_YOLO_ROS2)

注意：

- 运行时可能报错 段错误(核心已转储)，这是你自己的opencv版本和ros默认的版本造成冲突导致的，删除自己的版本使用ros默认的opencv即可解决报错
- ~~目前仅支持TensorRT 8，使用10会报错~~

## 文件结构

tensorrt_yolo/ 
 
├── CMakeLists.txt 

├── images 

│   ├── bus.jpg 

│   ├── dog.jpg 

│   ├── eagle.jpg 

│   ├── field.jpg 

│   ├── giraffe.jpg 

│   ├── herd_of_horses.jpg 

│   ├── person.jpg 

│   ├── room.jpg 

│   ├── street.jpg 

│   └── zidane.jpg 

├── include 

│   ├── bytekalman_filter.h 

│   ├── byte_tracker.h 

│   ├── calibrator.h 

│   ├── camera_infer.h 

│   ├── dataType.h 
 
│   ├── image_infer.h 

│   ├── infer.h 

│   ├── InferResult.h 

│   ├── KeyPoint.h 

│   ├── lapjv.h 

│   ├── postprocess.h 

│   ├── preprocess.h 

│   ├── public.h 

│   ├── Results.h 

│   ├── strack.h 

│   ├── types.h 

│   └── utils.h 

├── launch 

│   ├── camera.launch 

│   ├── d435i_yolo.launch 

│   └── modeltrans.launch 

├── msg 

│   ├── InferResult.msg 

│   ├── KeyPoint.msg 

│   └── Results.msg 

├── onnx_model 

│   └── yolov8s.onnx 
 
├── package.xml 

├── README-en.md 

├── README.md 

└── src 

   ├── bytekalman_filter.cpp 

   ├── byte_tracker.cpp 

   ├── calibrator.cpp 

   ├── camera_infer.cpp 

   ├── image_infer.cpp 

   ├── infer.cpp 

   ├── infer_get.cpp 

   ├── lapjv.cpp 
 
   ├── postprocess.cu 

   ├── preprocess.cu 

   └── strack.cpp 
 

6 directories, 49 files


## 导出ONNX模型

1. 安装 `YOLOv8`

```bash
pip install ultralytics
```

- 建议同时从 `GitHub` 上 clone 或下载一份 `YOLOv8` 源码到本地；
- 在本地 `YOLOv8`一级 `ultralytics` 目录下，新建 `weights` 目录，并且放入`.pt`模型

2. 安装onnx相关库

```bash
pip install onnx==1.12.0
pip install onnxsim==0.4.33
```

3. 导出onnx模型

- 可以在一级 `ultralytics` 目录下，新建 `export_onnx.py` 文件
- 向文件中写入如下内容：

```python
from ultralytics import YOLO

model = YOLO("./weights/Your.pt", task="detect")
path = model.export(format="onnx", simplify=True, device=0, opset=12, dynamic=False, imgsz=640)
```

- 运行 `python export_onnx.py` 后，会在 `weights` 目录下生成 `.onnx`

## 安装编译

1. 将仓库clone到自己的ros工作空间中；

   ```bash
   cd catkin_ws/src
   git clone git@github.com:gitagitty/tensorrt_yolo.git
   ```

2. 如果是自己数据集上训练得到的模型，记得更改 `launch` 中的相关配置，所有的配置信息全部都包含在`launch`文件中；

3. 确认 `CMakeLists.txt` 文件中 `cuda` 和 `tensorrt` 库的路径，与自己环境要对应，一般情况下是不需修改的；

4. 将已导出的 `onnx` 模型拷贝到 `onnx_model` 目录下

5. 编译工作空间

## 运行节点

必须先运行launch文件，再运行节点
节点 image_infer_node用于images文件夹内图片推理

d435i相机可以直接运行launch文件

```bash
roslaunch tensorrt_yolo d435i_yolo.launch 
```
通用launch文件
```
roslaunch tensorrt_yolo camera.launch
```
在launch文件中调整自己的参数，需要移植可以直接将下面这部分复制到自己的launch文件当中

```xml
    <!-- 启动目标检测节点 -->
    <node name="yolo_node" pkg="tensorrt_yolo" type="camera_infer_node" output="screen">

        <!-- 是否启动目标跟踪 -->
        <param name="track" value="false"/>
        <!-- 是否启动深度相机 -->
        <param name="depth" value="false"/>
        <!-- 是否启动姿态检测 -->
        <param name="pose" value="false"/>

        <!-- rgb图像topic -->
        <param name="rgbImageTopic" value="camera/color/image_raw"/>
        <!-- depth图像订阅地址,没有则忽略 -->
        <param name="depthImageTopic" value="/camera/depth/image_rect_raw"/>

        <!-- .plan文件地址 -->
        <param name="planFile" value="/home/evan/catkin_ws/src/tensorrt_yolo/onnx_model/yolov8s.plan"/>
        <!-- .onnx文件地址 -->
        <param name="onnxFile" value="/home/evan/catkin_ws/src/tensorrt_yolo/onnx_model/yolov8s.onnx"/>

        <!-- 非极大值抑制 -->
        <param name="nmsThresh" type = "double" value="0.7"/>
        <!-- 置信度 -->
        <param name="confThresh" type = "double" value="0.7"/>

        <!-- 目标检测类型数量 -->
        <param name="numClass" type = "int" value="1"/>
        <!-- 姿态检测特征点 -->
        <param name="numKpt" type = "int" value="17"/>
        <!-- 姿态检测维度 -->
        <param name="kptDims" type = "int" value="3"/>

        <!-- 推理模式选择 -->
        <!-- fp16 -->
        <param name="bFP16Mode" type = "bool" value="true"/>
        <!-- int8 -->
        <param name="bINT8Mode" type = "bool" value="false"/>

   </node> 
   
   <!-- 图片推理文件夹 -->
   <param name="imageDir" value="/home/evan/catkin_ws/src/tensorrt_yolo/images"/> 
   
   <!-- 目标检测类型名称 -->
   <param name="vClassNames" type="yaml" value="[ 'person' ]"/>   
```

## 节点订阅数据

```
const std::string rgbImageTopic = "/camera/color/image_raw"; //ros中image的topic
const std::string depthImageTopic = "/camera/depth/image_rect_raw"; //ros中depth image的topic
```

## 节点发布数据

节点对外publisher有一个，为infer_results，发布的内容均为一个列表，列表中的元素结构是

```cpp
float32[4] bbox
float32 conf
int32 classId
float32[3] coordinate //d_camera_infer_node发布三维空间坐标，camera_infer_node发布二维坐标，z=0
int32 Id // 关闭追踪模式默认为0，开启目标追踪为当前追踪的id值
float32 kpts // 存储未缩放到原始图像上的关键点数据
KeyPoint[] kpts // 存储经过缩放处理后的关键点数据，为了将关键点坐标映射回原始图像中的坐标系

```



