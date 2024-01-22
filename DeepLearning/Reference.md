# 01. 卷积和BN如何融合提升推理速度

- Conv和BN的融合：在网络的推理阶段，可以将BN层的运算融合到Conv层中，减少运算量，加速推理。本质上是修改了卷积核的参数，在不增加Conv层计算量的同时，略去了BN层的计算量。

![Alt](assert/fuse_conv_bn.png#pic_center=600x400)

```python
def fuse_conv_bn(conv, bn):

    std = (bn.running_var + bn.eps).sqrt()
    bias = bn.bias - bn.running_mean * bn.weight / std

    t = (bn.weight / std).reshape(-1, 1, 1, 1)
    weights = conv.weight * t

    conv = nn.Conv2d(3，128，3)
    conv.weight = torch.nn.Parameter(weights)
    conv.bias = torch.nn.Parameter(bias)
    return conv
```

# 02. 多卡BN如何处理（Synchronize BN）

- 首先解释下为什么需要同步 BN 操作，在多卡训练时，针对某些 BS 较小的任务（比如实例分割）每张卡计算得到的统计量可能与整体数据样本具有较大差异。换言之，针对具有大 BS 的分类任务，在训练阶段就无需使用多卡同步BN。
- 然后回答下多卡 BN 的要同步哪些东西，回想下多卡BN里面需要计算的参数均值和方差，多卡同步就是同步每张卡上对应的 BN 层分别计算出相应的统计量。
- 再来回答下多卡同步，跨卡同步BN的关键是在前向运算的时候拿到全局的均值和方差，在后向运算时候得到相应的全局梯度。最简单的实现方法是先同步求均值，再发回各卡然后同步求方差，但是这样就同步了两次。

![Alt](assert/syncbn.jpg#pic_center=600x400)

详细参考：https://zhuanlan.zhihu.com/p/40496177


# 03. TensorRT为什么能让模型跑更快

- 首先，TRT实现对于网络结构的垂直整合，即将目前主流神经网络的Conv、BN、Relu三个层融合为了一个层，所谓CBR

- 然后，TRT可以对网络结构做水平组合，水平组合是指将输入为相同张量和执行相同操作的层融合一起，比如将三个相连的1×1的CBR为一个大的1×1的CBR。

- 最后，对于concat层，将contact层的输入直接送入下面的操作中，不用单独进行concat后在输入计算，相当于减少了一次传输吞吐。

详细参考：https://zhuanlan.zhihu.com/p/64933639

# 04. 损失函数的应用-合页损失

<b><details><summary>SVM分类器</summary></b>
    
    - SVM是为了找到具有最大间隔的超平面，损失函数被称为 hinge loss，用于最大间隔分类，公式为 l = max（0 ，1 - z）， 其中 z 为 y（wx+b）， hinge loss 保证了所有普通样本（非支持向量的样本损失为0）不参与决定最终的超平面，减少了对训练样本的依赖程度，这就是为啥svm训练效率这么高。

</details>

# 05.Pytorch DataLoader的主要参数有哪些

- 其中最重要的参数dataset，代表传入的数据集，数据集需要包含__init__, __len__, __getitem__ 三个方法

详细参考：https://www.zdaiot.com/MLFrameworks/Pytorch/Pytorch%20DataLoader%E8%AF%A6%E8%A7%A3/

# 06.神经网络引入注意力机制后效果降低的原因

第一个角度是模型的欠拟合与过拟合；大部分注意力模块是有参数的，添加注意力模块会导致模型的复杂度增加。如果添加attention前模型处于欠拟合状态，那么增加参数是有利于模型学习的，性能会提高。

# 07. 为什么交叉熵可以作为损失函数

我们希望模型学到的分布和训练数据的分布相同，即希望最小化KL散度(一般用于计算两个分布之间的不同，KL散度=交叉熵-熵)，而当训练数据分布是固定的时候，最小化KL散度等价于最小化交叉熵。

从另外一个角度上来理解，如果使用传统的平方误差损失的话，得到的代价函数不是一个凸函数，会有很多的局部最优点；而交叉熵函数会得到一个相对更加平滑的代价曲线。

# 08. 优化算法之异同 SGD/AdaGrad/Adam

- 参考链接：https://zhuanlan.zhihu.com/p/32230623


# 09. 有哪些权重初始化的方法

-  Xavier 只能针对类似 sigmoid 和 tanh 之类的饱和激活函数，而无法应用于 ReLU 之类的非饱和激活函数。a = np.sqrt(3/self.neurals)

- KaimingInit 将 weight 以 Kaiming 的方式初始化，将 bias 初始化成指定常量，通常用于初始化卷积，Kaiming初始化建议初始化每层权值为一个均值为0标准差为2 n l \sqrt{\frac{2}{n_l}} 
n 的高斯分布，并且偏差为0。

- 参考链接：https://zhuanlan.zhihu.com/p/148034113

- 初始化权值可能会对训练过程有什么影响？为什么初始化权值要以方差作为衡量条件？
我认为网络学习的是训练数据的空间分布，即训练收敛时，整个输出空间应该是输入空间分布的某种稳定投影。从层的角度来看，假如2层网络：A->B，B希望获得稳定输出，但由于每次学习更新导致A也在变化，所以B想稳定就比较难。怎么办，保证A和B的分布一样，这样学习就简单一点，即可以理解成信息流通更流畅。

# 10. MMengine的一些特性

- 可以通过指定 init_cfg=dict(type='Pretrained', checkpoint='path/to/ckpt') 来加载预训练权重

- 对不同层进行初始化

```python
# 对卷积做 Kaiming 初始化，线性层做 Xavier 初始化
toy_net = ToyNet(
    init_cfg=[
        dict(type='Kaiming', layer='Conv2d'),
        dict(type='Xavier', layer='Linear')
    ], )
toy_net.init_weights()
```

# 11. Modules的一些属性问题

```python
def children(self):
# model.children():每一次迭代返回的每一个元素实际上是 Sequential 类型,而Sequential类型又可以使用下标index索引来获取每一个Sequenrial 里面的具体层，比如conv层、dense层等；
def named_children(self):
# 每一次迭代返回的每一个元素实际上是 一个元组类型，元组的第一个元素是名称，第二个元素就是对应的层或者是Sequential。
def modules(self):
# 将整个模型的所有构成（包括包装层、单独的层、自定义层等）由浅入深依次遍历出来，只不过modules()返回的每一个元素是直接返回的层对象本身，
def named_modules(self, memo=None, prefix=''):
# named_modules()返回的每一个元素是一个元组，第一个元素是名称，第二个元素才是层对象本身。
>>> net = torch.nn.Linear(2, 2)
>>> net.state_dict()
OrderedDict([('weight', tensor([[-0.3558,  0.2153],
        [-0.2785,  0.6982]])), ('bias', tensor([ 0.5771, -0.6232]))])
>>> net.state_dict().keys()
odict_keys(['weight', 'bias'])
# 函数state_dict的作用是返回一个包含module的所有state的dictionary，而这个字典的Keys对应的就是parameter和buffer的名字names
# 函数load_state_dict的作用和上边介绍的state_dict的作用刚好相反，是将parameter和buffer加载到Module及其SubModule中去。
def parameters(self, recurse=True):
    for name, param in self.named_parameters(recurse=recurse):
        yield param
# 来遍历网络模型中的参数
```

- 参考链接：https://zhuanlan.zhihu.com/p/156127643


# 12. 激活函数的对比与优缺点

- Sigmoid函数饱和使梯度消失。当神经元的激活在接近0或1处时会饱和，在这些区域梯度几乎为0，这就会导致梯度消失，几乎就有没有信号通过神经传回上一层。

- Tanh解决了Sigmoid的输出是不是零中心的问题，但仍然存在饱和问题。为了防止饱和，现在主流的做法会在激活函数前多做一步batch normalization，尽可能保证每一层网络的输入具有均值较小的、零中心的分布。

- ReLU对于随机梯度下降的收敛有巨大的加速作用；sigmoid和tanh在求导时含有指数运算，而ReLU求导几乎不存在任何计算量。单侧抑制，相对宽阔的激活边界；

- Softsign函数表达式：f=min(max(0,x),6),特点：ReLU6 就是普通的 ReLU 但是限制最大输出值为6（对输出值做clip），这是为了在移动端设备float16的低精度的时候，也能有很好的数值分辨率，如果对 ReLU 的激活范围不加限制，输出范围为 0 到正无穷，如果激活值非常大，分布在一个很大的范围内，则低精度的float16无法很好地精确描述如此大范围的数值，带来精度损失。

- SoftPlus 可以作为 ReLu 的一个不错的替代选择，可以看到与 ReLU 不同的是，SoftPlus 的导数是连续的、非零的、无处不在的，这一特性可以防止出现 ReLU 中的 “神经元死亡” 现象。

- 参考链接：http://giantpandacv.com/academic/%E7%AE%97%E6%B3%95%E7%A7%91%E6%99%AE/%E7%BB%BC%E8%BF%B0%E7%B1%BB/%E7%9B%98%E7%82%B9%E5%BD%93%E5%89%8D%E6%9C%80%E6%B5%81%E8%A1%8C%E7%9A%84%E6%BF%80%E6%B4%BB%E5%87%BD%E6%95%B0%E5%8F%8A%E9%80%89%E6%8B%A9%E7%BB%8F%E9%AA%8C/


# 13. Transformer/CNN/RNN的时间复杂度对比

- https://zhuanlan.zhihu.com/p/264749298



# 14.

- https://zhuanlan.zhihu.com/p/70703846

- https://zhuanlan.zhihu.com/p/51566209

ShuffleNet v1中提出的通道洗牌（Channel Shuffle）操作非常具有创新点，其对于解决分组卷积中通道通信困难上非常简单高效。

ShuffleNet v2分析了模型性能更直接的指标：运行时间。通道分割也是创新点满满。通过仔细分析通道分割，我们发现了它和DenseNet有异曲同工之妙，在这里轻量模型和高精度模型交汇在了一起。

![Alt](assert/shuffle.png#pic_center=600x400)

```python
def channel_shuffle(x, groups):
    """
    Parameters
        x: Input tensor of with `channels_last` data format
        groups: int number of groups per channel
    Returns
        channel shuffled output tensor
    Examples
        Example for a 1D Array with 3 groups
        >>> d = np.array([0,1,2,3,4,5,6,7,8])
        >>> x = np.reshape(d, (3,3))
        >>> x = np.transpose(x, [1,0])
        >>> x = np.reshape(x, (9,))
        '[0 1 2 3 4 5 6 7 8] --> [0 3 6 1 4 7 2 5 8]'
    """
    height, width, in_channels = x.shape.as_list()[1:]
    channels_per_group = in_channels // groups
    x = K.reshape(x, [-1, height, width, groups, channels_per_group])
    x = K.permute_dimensions(x, (0, 1, 2, 4, 3))  # transpose
    x = K.reshape(x, [-1, height, width, in_channels])
    return x
```

# 15. 比较CNN和多层感知机MLP

- MLP由全连接层构成，每个神经元都和上一层中的所有节点连接，存在参数冗余；相比之下，CNN由于权重共享，参数更少，方便网络的训练与设计深层网络；
- MLP只接受向量输入，会丢失像素间的空间信息；CNN接受矩阵和向量输入，能利用像素间的空间关系
- MLP是CNN的一个特例，当CNN卷积核大小与输入大小相同时其计算过程等价于MLP

# 16. MMCV中Hook机制简介及创建一个新的Hook

- Runner是一个模型训练的工厂，HOOK可以理解为一种触发器，也可以理解为一种训练框架的架构规范，它规定了在算法训练过程中的种种操作，并且我们可以通过继承HOOK类，然后注册HOOK自定义我们想要的操作。
- MMCV在./mmcv/runner/hooks/hook.py中定义了Hook的基类以及Hook的注册器HOOKS。作为基类，Hook本身没有实现具体的函数，只是提供了before_run、after_run等6个接口函数，其他所有的Hooks都通过继承Hook类并重写相应的函数完整指定功能。
- Hook 的主要目的是扩展功能，而不是修改已经实现的功能。如果我们实现一个定制化的 Hook，使用时需要定义、注册、调用3个步骤。自定义Hook在./mmcv/runner/hooks目录下构建，在执行runner.run()前会调用BaseRunner类中的register_training_hooks方法进行注册，使用build_from_cfg进行实例获取，然后调用BaseRunner类的register_hook()进行注册，这样所有Hook实例就都被纳入到runner中的一个list中。在runner执行过程中，会在特定的程序位点通过call_hook()函数调用相应的Hook。

```python
import torch
from mmcv.runner.hooks import HOOKS, Hook

@HOOKS.register_module()
class CheckInvalidLossHook(Hook):
    def __init__(self, interval=50):
        self.interval = interval

    def after_train_iter(self, runner):
        if self.every_n_iters(runner, self.interval):
            assert torch.isfinite(runner.outputs['loss']), \
                runner.logger.info('loss become infinite or NaN!')

def register_checkpoint_hook(self, checkpoint_config):
    hook = mmcv.build_from_cfg(checkpoint_config, HOOKS)
    self.register_hook(hook, priority='NORMAL')
```

# 17. 深度学习训练中如何区分错误样本和难例样本

- 一种方式是通过损失处理，论文标题：Unsupervised Label Noise Modeling and Loss Correction，可以使用一个Beta分布来刻画正常样本和噪音样本，从而将二者区分。


# 18. PyTorch 节省显存的常用策略

- 混合精度训练
- 大 batch 训练或者称为梯度累加：具体实现是在 loss = loss / cumulative_iters
- gradient checkpointing 梯度检查点

# 19. 深度学习模型训练时的Warmup预热学习率作用

Warmup是在ResNet论文中提到的一种学习率预热的方法，它在训练开始的时候先选择使用一个较小的学习率，训练了一些epoches或者steps(比如4个epoches,10000steps)，再修改为预先设置的学习来进行训练。

由于刚开始训练时，模型的权重(weights)是随机初始化的，此时若选择一个较大的学习率，可能带来模型的不稳定(振荡)，选择Warmup预热学习率的方式，可以使得开始训练的几个epoches或者一些steps内学习率较小，在预热的小学习率下，模型可以慢慢趋于稳定，等模型相对稳定后再选择预先设置的学习率进行训练，使得模型收敛速度变得更快，模型效果更佳。


# 21. PyTorch中的 ModuleList 和 Sequential的区别和使用场景

[ModuleList](https://zhuanlan.zhihu.com/p/64990232)

# 22. 考虑一个过滤器[-1 -1 -1; 0 0 0; 1 1 1] 用于卷积。该滤波器将从输入图像中提取哪些边缘？

该过滤器将从图像中提取水平边缘。为了获得更具体的理解，请考虑由具有以下像素强度的数组表示的灰度图像： 

```
[0 0 0 0 0 0; 
 0 0 0 0 0 0; 
 0 0 0 0 0 0; 
 10 10 10 10 10 10；
 10 10 10 10 10 10；]
```

从阵列中可以明显看出，图像的上半部分是黑色的，而下半部分是较浅的颜色，在图像中心形成明显的边缘。
两者的卷积将得到数组 [0 0 0 0; 30 30 30 30；30 30 30 30；0 0 0 0;]。从结果数组中的值可以看出，水平边缘已被识别。

# 23. 深度学习中为什么不对 bias 偏置进行正则化？

因为它对输入参数不敏感 公式上看就是它对所有的输入一视同仁，不贡献模型的曲率，求导的时候 bias 没多大作用

# 25. 深度学习模型中如何融入传统图像处理的特征？直接拼接融合有什么问题？

特征融合的一大难点在于不同的特征来自不同domain，直接物理拼接可能没有意义。比如常见的前融合或者后融合，以后融合举例，在卷积层铺平后的向量与传统视觉特征向量拼接，然后再接到全连接网络，可能效果不理想。

一种思路是采用discrimination correlation analysis方法进行融合，具体的：利用训练好的网络，从训练图像中提取特征向量;同时利用sift或者其他特征提取方式从训练图像中提取特征向量。然后对两组向量做dca分析，采用向量拼接的方式连接两组向量，训练分类器。

# 26. 多任务学习中各个任务损失的权重应该如何设计呢？

[多任务学习损失](https://www.zhihu.com/question/359962155)
[多任务学习学习率](https://zhuanlan.zhihu.com/p/56613537)

# 27. 为什么Adam常常打不过SGD？症结点与改善方案？

Adam拥有收敛速度快、调参容易的优点，却也存在时常被人攻击的泛化性与收敛问题。讨论模型泛化能力时，我们会希望模型找到的minimum是一个比较平缓 (flat) 、而非陡峭 (sharp) 的位置。不过相当多的论文指出或讨论了Adam在testing时error较大的事实。

![Alt](assert/sgd.jpg#pic_center=600x400)

[为什么Adam常常打不过SGD](https://medium.com/ai-blog-tw/deep-learning-%E7%82%BA%E4%BB%80%E9%BA%BCadam%E5%B8%B8%E5%B8%B8%E6%89%93%E4%B8%8D%E9%81%8Esgd-%E7%99%A5%E7%B5%90%E9%BB%9E%E8%88%87%E6%94%B9%E5%96%84%E6%96%B9%E6%A1%88-fd514176f805)

#  28. 如何处理不平衡的数据集？

有多种方法可以处理不平衡的数据集，例如使用不同的算法、对类别进行加权或对少数类别进行过采样。

算法选择：某些算法比其他算法更适合处理不平衡数据。例如，决策树和随机森林往往在不平衡数据上表现良好，而逻辑回归或支持向量机等算法可能会很困难。

类权重：通过为少数类分配更高的权重，可以使算法在训练过程中更加重视它。这有助于防止算法总是预测多数类。

过采样：您可以通过随机复制现有样本或基于现有样本生成新样本来创建少数类的合成样本。这可以平衡类别分布并帮助算法更多地了解少数类别。

# 29. Pytorch代码中如何尽量避免.to(device)的操作？

- 其中一种情况是初始化一个全0或全1的张量，比如模型的输出已经在cuda上了，你需要另外的tensor也是在cuda上，这时，你可以使用*_like操作符.

# 30. Pytorch中nn.Identity()/torch.chunk/torch.masked_select/torch.gather操作的应用场景？

```python
# 1:1的映射替换一些层
model = resnet50(pretrained=True)
model.fc = nn.Identity()

# 将输出分成N块
o1, o2, o3 = torch.chunk(one_layer(batch), 3, dim=1)

# 计算损失只在满足某些条件的张量上
data = torch.rand((3, 3)).requires_grad_()
mask = data > data.mean()
torch.masked_select(data, mask)
tensor([[0.0582, 0.7170, 0.7713],
        [0.9458, 0.2597, 0.6711],
        [0.2828, 0.2232, 0.1981]], requires_grad=True)
```