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
