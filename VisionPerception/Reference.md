# 01. 人脸识别任务中，ArcFace为什么比CosFace效果好

- 首先，ArcFace 和 CosFace 都是用于人脸识别的损失函数，它们的目标都是增强类间差异，减小类内差异，从而提高人脸识别的准确性。
- 其次，ArcFace是直接在角度空间θ中最大化分类界限，而CosFace是在余弦空间cos(θ)中最大化分类界限。角度间隔比余弦间隔在对角度的影响更加直接。

![Alt](assert/arcface.png#pic_center=600x400)

- ArcFace对特征向量归一化和加性角度间隔，提高了类间可分性同时加强类内紧度和类间差异，Pytorch实现如下：

```python
class ArcFace(nn.Module):
    def __init__(self, cin, cout, s=8, m=0.5):
        super().__init__()
        self.s = s
        self.sin_m = torch.sin(torch.tensor(m))
        self.cos_m = torch.cos(torch.tensor(m))
        self.cout = cout
        self.fc = nn.Linear(cin, cout, bias=False)

    def forward(self, x, label=None):
        # 计算权重向量的L2范数
        w_L2 = linalg.norm(self.fc.weight.detach(), dim=1, keepdim=True).T
        # 计算输入特征向量的L2范数
        x_L2 = linalg.norm(x, dim=1, keepdim=True)
        # 计算余弦相似度
        cos = self.fc(x) / (x_L2 * w_L2)

        if label is not None:
            sin_m, cos_m = self.sin_m, self.cos_m
            # 对标签进行one-hot编码
            one_hot = F.one_hot(label, num_classes=self.cout)
            # 计算sin和cos
            sin = (1 - cos ** 2) ** 0.5
            # 计算角度和
            angle_sum = cos * cos_m - sin * sin_m
            # 根据标签应用角度间隔
            cos = angle_sum * one_hot + cos * (1 - one_hot)
            # 缩放特征向量
            cos = cos * self.s
                        
        return cos
```

# 02. FCOS如何解决重叠样本，以及centerness的作用

- 第一个问题是FCOS预测什么，其实同anchor的预测一样，预测一个4D的向量（l,r,b,t）和类别；
- 如何确认正负样本，如果一个location(x, y)落到了任何一个GT box中，那么它就为正样本，这样相对于anchor系列的正样本就会多很多；
- 如何处理一个location(x, y)对应多个GT，分两步：首先利用FPN进行多尺度预测，不太层负责不同大小的目标；然后再取最小的那个当做回归目标；
- 由于我们把中心点的区域扩大到整个物体的边框，经过模型优化后可能有很多中心离GT box中心很远的预测框，为了增加更强的约束。当loss越小时，centerness就越接近1，也就是说回归框的中心越接近真实框；
- FCOS在处理遮挡和尺度变化问题上具有优势；

# 03. Centernet为什么可以去除NMS，以及正负样本的定义

- 采用下采样代替NMS，检测当前热点的值是否比周围的八个近邻点都大，然后取100个这样的点，采用的方式是一个3x3的MaxPool，类似于anchor-based检测中nms的效果；
- 模型输出是什么？模型的最后都是加了三个网络构造来输出预测值，默认是80个类、2个预测的中心点坐标、2个中心点的偏置，对应offset，scale，hm三个头；
- 如何构建正负样本？首先根据GT求中心点坐标，其次除以下采样倍率，然后用一个高斯核来将关键点分布到特征图上，中心位置是1，其余部位逐渐降低，其中方差是一个与目标大小(也就是w和h)相关的标准差。如果某一个类的两个高斯分布发生了重叠，直接取元素间最大的就可以。在CenterNet中，每个中心点对应一个目标的位置，不需要进行overlap的判断。那么怎么去减少negative center pointer的比例呢？CenterNet是采用Focal Loss的思想。
- 重点看一下中心点预测的损失函数。当预测为1时候，

![Alt](assert/centernet.jpg#pic_center=600x400)

参考链接：https://zhuanlan.zhihu.com/p/66048276

# 04. 介绍CBAM注意力

- 卷积模块的注意力机制模块，是一种结合了空间（spatial）和通道（channel）的注意力机制模块。
- 单就一张图来说，通道注意力，关注的是这张图上哪些内容是有重要作用的。平均值池化对特征图上的每一个像素点都有反馈，而最大值池化在进行梯度反向传播计算时，只有特征图中响应最大的地方有梯度的反馈。
- 将 CBAM 集成到 ResNet 中的方式是在每个block之间；

![Alt](assert/cbam.png#pic_center=600x400)

# 05. 数据增强Mixup及其变体

- 在图像分类任务中，Mixup的核心思想是以某个比例将一张图像与另外一张图像进行线性混合，同时，以相同的比例来混合这两张图像的one-hot标签。
- y是one-hot标签，⽐如yi的标签为[0,0,1]，yj的标签为[1,0,0]，此时lambda为0.2，那么此时的标签就变为0.2*[0,0,1] + 0.8*[1,0,0] = [0.8,0,0.2]，其实Mixup的⽴意很简单，就是通过这种混合的模型来增强模型的泛化性；
- 在目标检测任务中，作者对Mixup进行了修改，在图像混合过程中，为了避免图像变形，保留了图像的几何形状，并将两个图像的标签合并为一个新的数组。
- 对两幅全局图像进行像素加权组合得到增强图像。下面的Mixup变体可以分为：全局图像混合，如：ManifoldMixup和Un-Mix；区域图像混合，如：CutMix、Puzzle-Mix、Attentive-CutMix和Saliency-Mix；

```python
def mixup(imgs, labels, sampler=np.random.beta, sampler_args=(1.5, 1.5)):
    """
    Mixup two images and their labels.

    Arguments:
        imgs (list or tuple) -- list of image array which to be mixed
        labels (list or tuple) -- list of labels corresponding to images
        sampler -- ratio sampler, default is beta distribution
        sampler_args (tuple) -- parameters of sampler, default is (1.5, 1.5)

    Returns:
        mix_img (3d numpy array) -- image after mixup
        mix_label (2d numpy array) -- label after mixup
    """
    assert len(imgs) == len(labels) == 2
    img1, img2 = imgs
    label1, label2 = labels
    
    # get ratio from sampler
    ratio = max(0, min(1, sampler(*sampler_args)))
    
    # mixup two images
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]
    h = max(h1, h2)
    w = max(w1, w2)
    mix_img = np.zeros((h, w, 3))
    mix_img[:h1, :w1] = img1 * ratio
    mix_img[:h2, :w2] += img2 * (1 - ratio)
    mix_img = mix_img.astype('uint8')
    
    # mixup two labels
    mix_label = np.vstack((label1, label2))
    
    return mix_img, mix_label
```

参考链接：https://zhuanlan.zhihu.com/p/141878389

# 07. Yolov5的Foucs层和Passthrough层有什么区别

Focus层原理和PassThrough层很类似。它采用切片操作把高分辨率的图片（特征图）拆分成多个低分辨率的图片/特征图，即隔列采样+拼接。

原始的640 × 640 × 3的图像输入Focus结构，采用切片（slice）操作，先变成320 × 320 × 12的特征图，拼接（Concat）后，再经过一次卷积（CBL(后期改为SiLU，即为CBS)）操作，最终变成320 × 320 × 64的特征图。

![Alt](assert/foucs.jpg#pic_center=600x400)

Focus层将w-h平面上的信息转换到通道维度，再通过3*3卷积的方式提取不同特征。采用这种方式可以减少下采样带来的信息损失 。

```python
class Focus(nn.Module):
    # Focus wh information into c-space
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):  # ch_in, ch_out, kernel, stride
        super().__init__()
        self.conv = Conv(c1 * 4, c2, k, s, p, g, act)
 
    def forward(self, x):  # x(b,c,w,h) -> y(b, 4c, w/2, h/2)
        return self.conv(torch.cat([x[..., ::2, ::2], x[..., 1::2, ::2], x[..., ::2, 1::2], x[..., 1::2, 1::2]], 1)) 
        # 图片被分为4块。x[..., ::2, ::2]即行按2叠加取，列也是，对应上面原理图的“1”块块）， x[..., 1::2, ::2]对应“3”块块，x[..., ::2, 1::2]指“2”块块，x[..., 1::2, 1::2]指“4”块块。都是每隔一个采样（采奇数列）。用cat连接这些采样图，生成通道数为12的特征图
```

# 06. Yolov5的一些相关细节

- Letterbox的操作及其应用，训练时没有采用缩减黑边的方式，还是采用传统填充的方式，即缩放到416*416大小。只是在测试，使用模型推理时，才采用缩减黑边的方式，提高目标检测，推理的速度。第一步计算长和宽的最小缩放比，第二步计算缩放后的尺寸，第三步计算最小填充；实际中使用了一次cv2.resize，一次cv2.copyMakeBorder；

# 07. Yolov5与Yolov4相比neck部分有什么不同

- Yolov5的Neck和Yolov4中一样，都采用FPN+PAN的结构，但是在Yolov4的Neck结构中，采用的都是普通的卷积操作。而Yolov5的Neck结构中，采用借鉴CSPnet设计的CSP2结构，加强网络特征融合的能力。

# 8. YOLO9000为什么可以检测9000个类？

- 采用了一种联合训练的方法，

- WordTree如何表达对象的类别？首先在训练集的构建方面：按照各个类别之间的从属关系建立一种树型结构WordTree，对于物体的标签，采用one-hot编码的形式，数据集中的每个物体的类别标签被组织成1个长度为9418的向量，向量中除了在WordTree中从该物体对应的名词到根节点的路径上出现的词对应的类别标号处为1，其余位置为0。

- 在训练的过程中，当网络遇到来自检测数据集的图片时，用完整的YOLOv2loss进行反向传播计算，当网络遇到来自分类数据集的图片时，只用分类部分的loss进行反向传播。在类别概率预测上使用层次softmax处理，是每个层次类别上分别使用softmax。

- 预测时如何确定一个WordTree所对应的对象？既然各节点预测的是条件概率，那么一个节点的绝对概率就是它到根节点路径上所有条件概率的乘积。从根节点开始向下遍历，对每一个节点，在它的所有子节点中，选择概率最大的那个（一个节点下面的所有子节点是互斥的），一直向下遍历直到某个节点的子节点概率低于设定的阈值（意味着很难确定它的下一层对象到底是哪个），或达到叶子节点，那么该节点就是该WordTree对应的对象。

- 层次化Softmax的pytorch参考链接：https://geek-docs.com/pytorch/pytorch-questions/241_pytorch_tensorflow_hierarchical_softmax_implementation.html

- 参考链接：https://zhuanlan.zhihu.com/p/47575929


# 08. DETR的检测算法的创新点

- DETR将目标检测看作一种set prediction问题，CNN提取基础特征，送入*Transformer做关系建模*，得到的输出通过*二分图匹配算法*与图片上的ground truth做匹配。

- 问题1：将CNN backbone输出的feature map转化为能够被Transformer Encoder处理的序列化数据的过程？维度压缩，序列化特征，结合位置编码；

- 问题2：object queries是什么？object queries是N个learnable embedding，训练刚开始时可以随机初始化。在训练过程中，因为需要生成不同的boxes，object queries会被迫使变得不同来反映位置信息，所以也可以称为leant positional encoding；

- 问题3：Loss如何设计？二分图匹配需要一一配对，前面输出的N个；只要定义好每对prediction box和image object匹配的cost；如果预测的prediction box类别和image object类别相同的概率越大（越小），或者两者的box差距越小（越大），配对的cost 越小（越大）。

- 参考链接：https://zhuanlan.zhihu.com/p/267156624
- 参考链接：https://www.idea.edu.cn/news/2907.html

- 怎么理解Query，在 DETR 中，Object query 是由 transformer 解码器产生的，它由一组预定义的向量组成，每个向量代表一个预测框。这些向量可以被视为检测模型的输出类别和空间信息的结合，其中类别信息用于区分不同的目标，而空间信息则描述了目标在图像中的位置。而 DETR 中使用 object query 向量代替 anchor box，可以在不依赖于预先设定 anchor box 的情况下进行目标检测，从而更好地适应不同大小和形状的目标。

![Alt](assert/detr.png#pic_center=600x400)

# 13. 目标检测中旋转框IOU的计算方式

- 旋转框的两种编码方式：a: 中心点，尺度，角度；b: 顶点；

- 暴力方式：每次让一个框不动，另一个框旋转到和不动框垂直，然后计算IoU，最终取最小值，代表性的PIoU Loss。

- 传入两个旋转矩形的四个顶点坐标，然后计算它们之间的重叠面积（Rotated IoU）

```python
import numpy as np

def rotated_iou(rect1, rect2):
    """
    计算旋转矩形之间的重叠面积（Rotated IoU）

    参数：
    rect1：第一个旋转矩形的四个顶点坐标，格式为[[x1, y1], [x2, y2], [x3, y3], [x4, y4]]
    rect2：第二个旋转矩形的四个顶点坐标，格式为[[x1, y1], [x2, y2], [x3, y3], [x4, y4]]

    返回值：
    旋转矩形之间的重叠面积（Rotated IoU）
    """

    # 计算两个旋转矩形的最小外接矩形的坐标和宽高
    x_min1, y_min1, x_max1, y_max1 = np.min(rect1[:, 0]), np.min(rect1[:, 1]), np.max(rect1[:, 0]), np.max(rect1[:, 1])
    x_min2, y_min2, x_max2, y_max2 = np.min(rect2[:, 0]), np.min(rect2[:, 1]), np.max(rect2[:, 0]), np.max(rect2[:, 1])
    w1, h1 = x_max1 - x_min1, y_max1 - y_min1
    w2, h2 = x_max2 - x_min2, y_max2 - y_min2

    # 计算两个旋转矩形的面积
    area1 = w1 * h1
    area2 = w2 * h2

    # 计算两个旋转矩形的交集的面积
    intersection = np.maximum(0, np.minimum(x_max1, x_max2) - np.maximum(x_min1, x_min2)) * \
                   np.maximum(0, np.minimum(y_max1, y_max2) - np.maximum(y_min1, y_min2))

    # 计算旋转矩形的并集的面积
    union = area1 + area2 - intersection

    # 计算旋转矩形之间的重叠面积（Rotated IoU）
    iou = intersection / union

    return iou
```

# 14. 局部注意力如何实现

- 局部注意力则把加权求和的范围缩小到给定窗口大小的局部范围。于是，local Attention如何确定局部范围成为关键。第一种是采用自注意力机制来建模query对的关系。第二种是对query-independent的全局上下文建模。

- 参考链接：https://zhuanlan.zhihu.com/p/64988633

- Non-local block想要计算出每一个位置特定的全局上下文，但是经过训练之后，全局上下文是不受位置依赖的。三步实现注意力机制：1）全局attention pooling：采用1x1卷积和softmax函数来获取attention权值，然后执行attention pooling来获得全局上下文特征。2）特征转换：采用1x1卷积；3）特征聚合：采用相加操作将全局上下文特征聚合到每个位置的特征上。

- 卷积只能对局部区域进行context modeling，导致感受野受限制，而Non-local和SENet实际上是对整个输入feature进行context modeling，感受野可以覆盖到整个输入feature上，这对于网络来说是一个有益的语义信息补充。另外，网络仅仅通过卷积堆叠来提取特征，其实可以认为是用同一个形式的函数来拟合输入，导致网络提取特征缺乏多样性，而Non-local和SENet正好增加了提取特征的多样性，弥补了多样性的不足。

# 15. 视觉任务中的长尾问题的常见解决方案

- 两种基本方法：重采样、重加权，其中长尾分类最优的Decoupling算法依赖于2-stage的分步训练，特征提取backbone需要在长尾分布下学，而classifier又需要re-balancing的学。

- 但上述方法的问题是需要在训练/学习之前，了解“未来”将要看到的数据分布，这显然不符合人类的学习模式，De-confound-TDE

- 参考链接：https://zhuanlan.zhihu.com/p/259569655


# 16. Yolov5中的objectness的作用

- 如果我们在训练中改变objectness的权重（self.hyp['obj']），我们是否仍然应该在检测中简单地将objectness分数和cls分数相乘？
  
- 如果是，如果我们设置“self.hyp[‘obj’]=0”会怎么样？通过这样做，在训练期间将不会控制客观性分数。

- 为什么objectness损失会随着图像大小而变化？其受到正样本和负样本之间极度不平衡的影响。当图像放大时，其中的对象数量保持不变，因此不平衡性增加（变得更糟）。损失增益将按比例进行补偿。

# 20. 目标检测设置很多不同的anchor，能否改善小目标及非正常尺寸目标的性能，除计算速度外还存在什么问题

感受野足够大时，被忽略的信息就较少。

目标检测任务中设置anchor要对齐感受野，anchor太大或者偏离感受野会对性能产生一定的影响。

增大感受野的方法：使用空洞卷积；使用池化层；增大卷积核。


# 23. 在目标Crowded的场景下，经常在两个真正目标中间会出现误检的原因?

两个Proposal和左面GT的IoU一样，但是显然绿色的更容易受到2号GT的影响，从而远离它真实的目标，这也是为什么crowded的情况中，误检经常出现在两个真正GT中间的原因。但是我们为了抑制这样的误检，又不能一味降低NMS的阈值，否则会对于真正靠近的情况错误抑制。

[谈谈深度学习目标检测中的遮挡问题](https://zhuanlan.zhihu.com/p/43655912)

# 24. 在Unet网络结构中，四次降采样对于分割网络到底是不是必须的？

[研习Unet](https://zhuanlan.zhihu.com/p/44958351)

# 25. 为什么UNet++可以被剪枝，怎么去决定剪多少？

[研习Unet](https://zhuanlan.zhihu.com/p/44958351)

# 26. 在A场景下进行目标的标记及训练，如何在B场景下取得好的效果？

域迁移方向

# 27. 如何修改Yolov5目标检测，从而实现旋转目标检测？

[Yolo旋转目标检测](https://yunyang1994.gitee.io/2021/04/20/%E4%BF%AE%E6%94%B9-YOLOv5-%E6%BA%90%E7%A0%81%E5%9C%A8-DOTAv1.5-%E9%81%A5%E6%84%9F%E6%95%B0%E6%8D%AE%E9%9B%86%E4%B8%8A%E8%BF%9B%E8%A1%8C%E6%97%8B%E8%BD%AC%E7%9B%AE%E6%A0%87%E6%A3%80%E6%B5%8B/)
