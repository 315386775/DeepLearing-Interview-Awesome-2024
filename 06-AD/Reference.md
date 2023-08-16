# 01. 相机内外参

-   相机有两个最基础的数据：内参(Instrinsics)和外参(Extrinsics)，内参主要描述的是相机的CCD/CMOS感光片尺寸/分辨率以及光学镜头的系数，外参主要描述的是相机在世界坐标系下的摆放位置和朝向角度。

- 参考：https://zhuanlan.zhihu.com/p/646310465

# 02. 坐标系的变换

- BEV训练数据集的世界坐标系, 比如nuScenes地图，它的世界坐标系是图片坐标系，原点在图片左下角，单位是米。数据集中会根据时间序列给出车辆的瞬时位置，也就是在这个图片上的XY。

- BEV里，这个Ego是特指车辆本身，它是用来描述摄像机/激光雷达（Lidar，light detection and ranging）/毫米波雷达（一般代码里就简称为Radar）/IMU在车身上的安装位置（单位默认都是米）和朝向角度，坐标原点一般是车身中间，外参（Extrinsics Matrix）主要就是描述这个坐标系的。

- 相机坐标系，坐标原点在CCD/CMOS感光片的中央，单位是像素，内参（Intrinsics Matrix）主要就是描述这个坐标系的。

- 照片坐标系，坐标原点在图片的左上角，单位是像素，横纵坐标轴一般不写成XY，而是uv。

- 照片中的像素位置转换到世界坐标系时，要经历：Image_to_Camera, Camera_to_Ego, Ego_to_World；Camera_to_Image通常就是Intrinsics参数矩阵，Ego_to_Camera就是Extrinsics参数矩阵。

# 03. 放射变换与逆投影变换分别是什么

- 仿射变换： 仿射变换是一种线性变换，保持了直线的平行性和比例关系。它可以用于将一个二维平面上的点映射到另一个二维平面上。仿射变换可以通过一个矩阵乘法和一个平移向量来表示。它包括平移、旋转、缩放和剪切等操作。在计算机视觉领域，仿射变换常用于图像的平移、旋转、缩放和仿射校正等操作。

- 逆投影变换： 逆投影变换是指通过相机内参和外参，将图像上的点投影到三维空间中的过程。它是相机成像过程的逆过程。逆投影变换可以用于将图像上的点转换为三维空间中的点坐标。逆投影变换的计算需要相机的内参矩阵、外参矩阵和图像上的点坐标。在计算机视觉和计算机图形学中，逆投影变换常用于三维重建、相机姿态估计和虚拟现实等应用。

```python
import numpy as np

# 定义相机内参和外参
K = np.array([[1000, 0, 500], [0, 1000, 300], [0, 0, 1]], dtype=np.float32)
R = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float32)
T = np.array([1, 2, 3], dtype=np.float32)

# 定义图像上的点坐标
uv = np.array([[200, 300], [400, 500]], dtype=np.float32)

# 计算逆投影变换
Rc2w_invK = np.linalg.inv(np.dot(R, K))
H = np.dot(Rc2w_invK, np.append(uv, np.ones((uv.shape[0], 1)), axis=1).T)
Pxyz = H * (T[2] / H[2]) - T[:2]

# 定义仿射变换矩阵
M = np.array([[1, 0, 100], [0, 1, 50]], dtype=np.float32)

# 进行仿射变换
output = cv2.warpAffine(image, M, (image.shape[1], image.shape[0]))
```