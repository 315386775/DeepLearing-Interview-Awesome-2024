# 01. 相机内外惨

-   相机有两个最基础的数据：内参(Instrinsics)和外参(Extrinsics)，内参主要描述的是相机的CCD/CMOS感光片尺寸/分辨率以及光学镜头的系数，外参主要描述的是相机在世界坐标系下的摆放位置和朝向角度。

- 参考：https://zhuanlan.zhihu.com/p/646310465

# 02. 坐标系的变换

- BEV训练数据集的世界坐标系, 比如nuScenes地图，它的世界坐标系是图片坐标系，原点在图片左下角，单位是米。数据集中会根据时间序列给出车辆的瞬时位置，也就是在这个图片上的XY。

- BEV里，这个Ego是特指车辆本身，它是用来描述摄像机/激光雷达（Lidar，light detection and ranging）/毫米波雷达（一般代码里就简称为Radar）/IMU在车身上的安装位置（单位默认都是米）和朝向角度，坐标原点一般是车身中间，外参（Extrinsics Matrix）主要就是描述这个坐标系的。

- 相机坐标系，坐标原点在CCD/CMOS感光片的中央，单位是像素，内参（Intrinsics Matrix）主要就是描述这个坐标系的。

- 照片坐标系，坐标原点在图片的左上角，单位是像素，横纵坐标轴一般不写成XY，而是uv。

- 照片中的像素位置转换到世界坐标系时，要经历：Image_to_Camera, Camera_to_Ego, Ego_to_World；Camera_to_Image通常就是Intrinsics参数矩阵，Ego_to_Camera就是Extrinsics参数矩阵。