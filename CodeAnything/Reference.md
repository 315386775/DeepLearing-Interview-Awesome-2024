# 01. Pytorch实现注意力机制、多头注意力与自注意力

```python
class ScaledDotProductAttention(nn.Module):
    """ Scaled Dot-Product Attention """

    def __init__(self, scale):
        super().__init__()

        self.scale = scale
        self.softmax = nn.Softmax(dim=2)

    def forward(self, q, k, v, mask=None):
        u = torch.bmm(q, k.transpose(1, 2)) # 1.Matmul
        u = u / self.scale # 2.Scale

        if mask is not None:
            u = u.masked_fill(mask, -np.inf) # 3.Mask

        attn = self.softmax(u) # 4.Softmax
        output = torch.bmm(attn, v) # 5.Output

        return attn, output

if __name__ == "__main__":
    n_q, n_k, n_v = 2, 4, 4
    d_q, d_k, d_v = 128, 128, 64

    q = torch.randn(batch, n_q, d_q)
    k = torch.randn(batch, n_k, d_k)
    v = torch.randn(batch, n_v, d_v)
    mask = torch.zeros(batch, n_q, n_k).bool()

    attention = ScaledDotProductAttention(scale=np.power(d_k, 0.5))
    attn, output = attention(q, k, v, mask=mask)

    print(attn)
    print(output)

```
```python
from math import sqrt

import torch
import torch.nn as nn

class MultiHeadSelfAttention(nn.Module):
    dim_in: int  # input dimension
    dim_k: int   # key and query dimension
    dim_v: int   # value dimension
    num_heads: int  # number of heads, for each head, dim_* = dim_* // num_heads

    def __init__(self, dim_in, dim_k, dim_v, num_heads=8):
        super(MultiHeadSelfAttention, self).__init__()
        assert dim_k % num_heads == 0 and dim_v % num_heads == 0, "dim_k and dim_v must be multiple of num_heads"
        self.dim_in = dim_in
        self.dim_k = dim_k
        self.dim_v = dim_v
        self.num_heads = num_heads
        self.linear_q = nn.Linear(dim_in, dim_k, bias=False)
        self.linear_k = nn.Linear(dim_in, dim_k, bias=False)
        self.linear_v = nn.Linear(dim_in, dim_v, bias=False)
        self._norm_fact = 1 / sqrt(dim_k // num_heads)

    def forward(self, x):
        # x: tensor of shape (batch, n, dim_in)
        batch, n, dim_in = x.shape
        assert dim_in == self.dim_in

        nh = self.num_heads
        dk = self.dim_k // nh  # dim_k of each head
        dv = self.dim_v // nh  # dim_v of each head

        q = self.linear_q(x).reshape(batch, n, nh, dk).transpose(1, 2)  # (batch, nh, n, dk)
        k = self.linear_k(x).reshape(batch, n, nh, dk).transpose(1, 2)  # (batch, nh, n, dk)
        v = self.linear_v(x).reshape(batch, n, nh, dv).transpose(1, 2)  # (batch, nh, n, dv)

        dist = torch.matmul(q, k.transpose(2, 3)) * self._norm_fact  # batch, nh, n, n
        dist = torch.softmax(dist, dim=-1)  # batch, nh, n, n

        att = torch.matmul(dist, v)  # batch, nh, n, dv
        att = att.transpose(1, 2).reshape(batch, n, self.dim_v)  # batch, n, dim_v

        return att
```

更详细请查阅[注意力,多头注意力,自注意力及Pytorch实现](https://zhuanlan.zhihu.com/p/366592542)

# 02. Numpy广播机制实现矩阵间L2距离的计算

- 在符合广播条件的前提下，广播机制会为尺寸较小的向量添加一个轴（广播轴），使其维度信息与较大向量的相同。

- 计算 m*2 的矩阵与 n * 2 的矩阵中，m*2 的每一行到 n*2 的两两之间欧氏距离。

```python
# L2 = sqrt((x1-x2)^2 + (y1-y2)^2 + (z1-z2)^2)

def L2_dist_1(cloud1, cloud2):
    m, n = len(cloud1), len(cloud2)
    # project 01
    # cloud1 = np.repeat(cloud1, n, axis=0) # (n*m,2)
    # cloud1 = np.reshape(cloud1, (m, n, -1)) # (m,n,2) (n,2)

    # project 02
    # cloud1 = cloud1[:, None, :] # (m,1,2)
    
    # project 03
    cloud1 = np.expand_dims(cloud1, 1)
    
    dist = np.sqrt(np.sum((cloud1 - cloud2)**2, axis=2))
    return dist
```

# 03. Conv2D卷积的Python和C++实现

```c++
cv::Mat_<float> spatialConvolution(const cv::Mat_<float>& src, const cv::Mat_<float>& kernel)
{
    Mat dst(src.rows,src.cols,src.type());

    Mat_<float> flipped_kernel; 
    flip(kernel, flipped_kernel, -1);

    const int dx = kernel.cols / 2;
    const int dy = kernel.rows / 2;

    for (int i = 0; i<src.rows; i++) 
    {
        for (int j = 0; j<src.cols; j++) 
        {
            float tmp = 0.0f;
            for (int k = 0; k<flipped_kernel.rows; k++) 
            {
              for (int l = 0; l<flipped_kernel.cols; l++) 
              {
                int x = j - dx + l;
                int y = i - dy + k;
                if (x >= 0 && x < src.cols && y >= 0 && y < src.rows)
                    tmp += src.at<float>(y, x) * flipped_kernel.at<float>(k, l);
              }
            }
            dst.at<float>(i, j) = saturate_cast<float>(tmp);
        }
    }
    return dst.clone();
}

cv::Mat convolution2D(cv::Mat& image, cv::Mat& kernel) {
    int image_height = image.rows;
    int image_width = image.cols;
    int kernel_height = kernel.rows;
    int kernel_width = kernel.cols;
    cv::Mat output(image_height - kernel_height + 1, image_width - kernel_width + 1, CV_32S);

    for (int i = 0; i < output.rows; i++) {
        for (int j = 0; j < output.cols; j++) {
            for (int k = 0; k < kernel_height; k++) {
                for (int l = 0; l < kernel_width; l++) {
                    output.at<int>(i, j) += image.at<int>(i + k, j + l) * kernel.at<int>(k, l);
                }
            }
        }
    }

    return output;
}

```

```python
def convolution2D(image, kernel):
    image_height, image_width = image.shape
    kernel_height, kernel_width = kernel.shape
    output = np.zeros((image_height - kernel_height + 1, image_width - kernel_width + 1))

    for i in range(output.shape[0]):
        for j in range(output.shape[1]):
            output[i, j] = np.sum(image[i:i+kernel_height, j:j+kernel_width] * kernel)

    return output
```

# 06. Numpy实现bbox_iou的计算

```python

def IoU(boxA,boxB):#x1,y1,x2,y2
    xA=max(boxA[0],boxB[0])
    yA=max(boxA[1],boxB[1])
    xB=min(boxA[2],boxB[2])
    yB=min(boxA[3],boxB[3])
    interArea=max(0,xB-xA+1)*max(0,yB-yA+1)
    boxAArea=(boxA[2]-boxA[0]+1)*(boxA[3]-boxA[1]+1)
    boxBArea=(boxB[2]-boxB[0]+1)*(boxB[3]-boxB[1]+1)
    iou=interArea/(boxAArea+boxBArea-interArea)
    return iou
#
def calculate_iou(bbox1, bbox2):
    # 计算bbox的面积
    area1 = (bbox1[:, 2] - bbox1[:, 0]) * (bbox1[:, 3] - bbox1[:, 1])
    area2 = (bbox2[:, 2] - bbox2[:, 0]) * (bbox2[:, 3] - bbox2[:, 1])
    # 换一种更高级的方式计算面积
    # area2 = np.prod(bbox2[:, 2:] - bbox2[:, :2], axis=1)
    
    # 计算交集的左上角坐标和右下角坐标
    lt = np.maximum(bbox1[:, None, :2], bbox2[:, :2]) # [m, n, 2]
    rb = np.minimum(bbox1[:, None, 2:], bbox2[:, 2:])
    
    # 计算交集面积
    wh = np.clip(rb - lt, a_min=0, a_max=None)
    inter = wh[:,:,0] * wh[:,:,1]
    
    # 计算并集面积
    union = area1[:, None] + area2 - inter
    
    return inter / union
```

# 05. Numpy实现Focalloss

![Alt](assert/focal.jpg#pic_center)

Focal loss其实就是相当于给不同的概率，不同的权重来调整loss，从而让模型更加注意区分错误样本和难区分的样本。

```python

import numpy as np

def multiclass_focal_log_loss(y_true, y_pred, class_weights = None, alpha = 0.5, gamma = 2):
    """
    Numpy version of the Focal Loss
    """
    pt = np.where(y_true == 1, y_pred, 1-y_pred)
    alpha_t = np.where(y_true == 1, alpha, 1-alpha)
    # FL = - alpha_t (1-pt)^gamma log(pt)
    focal_loss = - alpha_t * (1 - pt) ** gamma * np.log(pt))
    if class_weights is None:
        focal_loss = np.mean(focal_loss)
    else:
        focal_loss = np.sum(np.multiply(focal_loss, class_weights))
    return focal_loss

# 示例用法
y_true = np.array([1, 0, 1, 0])
y_pred = np.array([0.9, 0.1, 0.8, 0.2])

loss = multiclass_focal_log_loss(y_true, y_pred)
print(loss)
```
# 06. Python实现nms、softnms

```python
def nms(bboxes, scores, iou_thresh):
    """
    :param bboxes: 检测框列表
    :param scores: 置信度列表
    :param iou_thresh: IOU阈值
    :return:
    """
    x1 = bboxes[:, 0]
    y1 = bboxes[:, 1]
    x2 = bboxes[:, 2]
    y2 = bboxes[:, 3]
    areas = (y2 - y1) * (x2 - x1)

    result = []
    index = scores.argsort()[::-1]  # 对检测框按照置信度进行从高到低的排序，并获取索引

    while index.size > 0:
        i = index[0]
        result.append(i)  # 将置信度最高的加入结果列表

        # 计算其他边界框与该边界框的IOU
        x11 = np.maximum(x1[i], x1[index[1:]])
        y11 = np.maximum(y1[i], y1[index[1:]])
        x22 = np.minimum(x2[i], x2[index[1:]])
        y22 = np.minimum(y2[i], y2[index[1:]])
        w = np.maximum(0, x22 - x11 + 1)
        h = np.maximum(0, y22 - y11 + 1)
        overlaps = w * h
        ious = overlaps / (areas[i] + areas[index[1:]] - overlaps)
        # 只保留满足IOU阈值的索引
        idx = np.where(ious <= iou_thresh)[0]
        index = index[idx + 1]  # 处理剩余的边框
    bboxes, scores = bboxes[result], scores[result]
    return bboxes, scores
```

要实现 Soft-NMS（软性非极大值抑制），需要对原始的 NMS 算法进行一些修改。Soft-NMS 通过逐渐降低重叠边界框的置信度，而不是直接将它们排除，从而更平滑地抑制重叠边界框的影响。

```python
def soft_nms(bboxes, scores, iou_thresh, sigma=0.5, score_thresh=0.001):
    x1 = bboxes[:, 0]
    y1 = bboxes[:, 1]
    x2 = bboxes[:, 2]
    y2 = bboxes[:, 3]
    areas = (y2 - y1) * (x2 - x1)

    for i in range(len(scores)):
        max_idx = i
        max_score = scores[i]

        # 与其他边界框计算IOU，并更新置信度
        for j in range(i + 1, len(scores)):
            if scores[j] > score_thresh:
                x11 = np.maximum(x1[i], x1[j])
                y11 = np.maximum(y1[i], y1[j])
                x22 = np.minimum(x2[i], x2[j])
                y22 = np.minimum(y2[i], y2[j])
                w = np.maximum(0, x22 - x11 + 1)
                h = np.maximum(0, y22 - y11 + 1)
                overlaps = w * h
                iou = overlaps / (areas[i] + areas[j] - overlaps)
                decay = np.exp(-(iou * iou) / sigma)
                scores[j] = scores[j] * decay

                # 保留置信度最高的边界框
                if scores[j] > max_score:
                    max_idx = j
                    max_score = scores[j]

        # 交换置信度最高的边界框和当前边界框的位置
        bboxes[i], bboxes[max_idx] = bboxes[max_idx], bboxes[i]
        scores[i], scores[max_idx] = scores[max_idx], scores[i]

    # 过滤置信度低于阈值的边界框
    selected_idx = np.where(scores > score_thresh)
    bboxes = bboxes[selected_idx]
    scores = scores[selected_idx]

    return bboxes, scores
```

# 07. Python实现BN批量归一化

实现BN需要求的：均值、方差、参数beta、参数gamma。

![Alt](assert/bn.png#pic_center)

```python
class MyBN:
    def __init__(self, momentum, eps, num_features):
        """
        初始化参数值
        :param momentum: 追踪样本整体均值和方差的动量
        :param eps: 防止数值计算错误
        :param num_features: 特征数量
        """
        # 对每个batch的mean和var进行追踪统计
        self._running_mean = 0
        self._running_var = 1
        self._momentum = momentum
        self._eps = eps
        # 对应论文中需要更新的beta和gamma，采用pytorch文档中的初始化值
        self._beta = np.zeros(shape=(num_features, ))
        self._gamma = np.ones(shape=(num_features, ))

    def batch_norm(self, x):
        x_mean = x.mean(axis=0)
        x_var = x.var(axis=0)
        # 对应running_mean的更新公式
        self._running_mean = (1-self._momentum)*x_mean + self._momentum*self._running_mean
        self._running_var = (1-self._momentum)*x_var + self._momentum*self._running_var
        # 对应论文中计算BN的公式
        x_hat = (x-x_mean)/np.sqrt(x_var+self._eps)
        y = self._gamma*x_hat + self._beta
        return y
```

更详细请查阅[BN](https://zhuanlan.zhihu.com/p/100672008)

# 10. PyTorch卷积与BatchNorm的融合

更详细请查阅[CONV-BN](https://zhuanlan.zhihu.com/p/49329030)


# 11. 分割网络损失函数Dice Loss代码实现

```python
# 防止分母为0
smooth = 100
 
# 定义Dice系数
def dice_coef(y_true, y_pred):
    y_truef = K.flatten(y_true)  # 将y_true拉为一维
    y_predf = K.flatten(y_pred)
    intersection = K.sum(y_truef * y_predf)
    return (2 * intersection + smooth) / (K.sum(y_truef) + K.sum(y_predf) + smooth)
 
# 定义Dice损失函数
def dice_coef_loss(y_true, y_pred):
    return 1-dice_coef(y_true, y_pred)
```

# 08. Pytorch 针对L1损失的输入需要做数值的截断，构建CustomL1Loss类

```python
class CustomL1Loss(nn.Module):
    def __init__(self, low=-128, high=128):
        super().__init__()
        self.low, self.high = low, high
        self.l1_loss = nn.SmoothL1Loss()

    def forward(self, output, target):
        output = torch.clip(output, min=self.low, max=self.high)
        target = torch.clip(target, min=self.low, max=self.high)
        return self.l1_loss(output, target)
```


# 12. Numpy实现一个函数来计算两个向量之间的余弦相似度

```python
import numpy as np

def cosine_similarity(vector1, vector2):
    dot_product = np.dot(vector1, vector2)
    magnitude1 = np.linalg.norm(vector1)
    magnitude2 = np.linalg.norm(vector2)
    return dot_product / (magnitude1 * magnitude2)
```

# 13. Numpy实现Sigmoid函数

```python
import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def softmax(x):
    shift_x = x - np.max(x)
    exp_x = np.exp(shift_x)
    return exp_x / np.sum(exp_x)
```

# 18. Numpy 完成稀疏矩阵的类，并实现add和multiply的操作

```python
class SparseMatrix:
    def __init__(self, matrix):
        self.matrix = matrix

    def add(self, other_matrix):
        result = []
        for i in range(len(self.matrix)):
            row = []
            for j in range(len(self.matrix[0])):
                row.append(self.matrix[i][j] + other_matrix.matrix[i][j])
            result.append(row)
        return SparseMatrix(result)

    def multiply(self, other_matrix):
        result = []
        for i in range(len(self.matrix)):
            row = []
            for j in range(len(other_matrix.matrix[0])):
                element = 0
                for k in range(len(self.matrix[0])):
                    element += self.matrix[i][k] * other_matrix.matrix[k][j]
                row.append(element)
            result.append(row)
        return SparseMatrix(result)

    def __str__(self):
        return '\n'.join([' '.join(map(str, row)) for row in self.matrix])

# 测试
matrix1 = SparseMatrix([[1, 0, 0], [0, 0, 2]])
matrix2 = SparseMatrix([[0, 0, 3], [0, 4, 0]])
```

# 08. Pytorch 实现SGD优化算法

```python
from torch import optim
from optimizers.misc import validator, Optimizer


class SGD(Optimizer):
    def __init__(self, params, lr):
        self.lr = lr
        super(SGD, self).__init__(params)

    def step(self):
        for p in self.params:
            if p.grad is not None:
                p.data -= self.lr * p.grad.data
```


# 111. C++中与类型转换相关的4个关键字特点及应用场合

```c++
static_cast<type_id> ()  // 主要用于C++内置基本类型之间的转换
const_cast<>()  //　用于将const类型的数据和非const类型的数据之间进行转换
dynamic_cast<>()  //　可以将基类对象指针(引用)cast到继承类指针，（类中必须有虚函数）
reinterpret_cast<>()  //　type_id必须是指针，引用...可以把一个指针与内置类型之间进行转换
```