# 01. C++中与类型转换相关的4个关键字特点及应用场合

```c++
static_cast<type_id> ()  // 主要用于C++内置基本类型之间的转换
const_cast<>()  //　用于将const类型的数据和非const类型的数据之间进行转换
dynamic_cast<>()  //　可以将基类对象指针(引用)cast到继承类指针，（类中必须有虚函数）
reinterpret_cast<>()  //　type_id必须是指针，引用...可以把一个指针与内置类型之间进行转换
```

# 02. Python装饰器及其作用

- 详细讲述链接：https://foofish.net/python-decorator.html

- Python中的函数可以像普通变量一样当做参数传递给另外一个函数，装饰器本质上是一个类/函数，它可以在不修改函数主体功能的前提下增加额外的功能，装饰器的返回值也是一个类对象或者函数，可以用来插入日志，性能测试等等，装饰器的使用提高了函数的复用性，增加了程序的可读性。

# 03. map,lambda,filter,reduce的用法

```python
##  lambda匿名函数
# lambda 参数:操作(参数)
add = lambda x, y: x + y
print(add(3, 5))
# 按照第二个数排序
a = [(1, 2), (4, 1), (9, 10), (13, -3)]
a.sort(key=lambda x: x[1])

# map会将一个函数映射到一个输入列表的所有元素上 map(func, list_of_inputs)
# func一般会结合lambdas函数食用
squared	= list(map(lambda x: x**2, lists))

# filter过滤列表中的元素,并且返回一个由所有符合要求的元素所构成的列表,符合要求即函数映射到该元素时返回值为True.	
less_than_zero = filter(lambda x: x<0, number_list)

from functools import reduce  # reduce() 函数会对参数序列中元素进行累计。
product	= reduce((lambda x, y:x*y),[1,2,3,4])  # 24

```

# 02. 用Numpy的广播机制实现矩阵之间距离的计算

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

# 04. Pytorch实现注意力机制、多头注意力

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

        # self attention
        # dist = torch.bmm(q, k.transpose(1, 2)) * self._norm_fact  # batch, n, n
        # dist = torch.softmax(dist, dim=-1)  # batch, n, n

        # att = torch.bmm(dist, v)

        return att

```

- 参考链接：https://zhuanlan.zhihu.com/p/366592542


# 05. C++实现Conv2D

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
#

def convolution2D(image, kernel):
    image_height, image_width = image.shape
    kernel_height, kernel_width = kernel.shape
    output = np.zeros((image_height - kernel_height + 1, image_width - kernel_width + 1))

    for i in range(output.shape[0]):
        for j in range(output.shape[1]):
            output[i, j] = np.sum(image[i:i+kernel_height, j:j+kernel_width] * kernel)

    return output


```