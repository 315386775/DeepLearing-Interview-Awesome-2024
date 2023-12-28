# **DeepLearning-Interview-Awesome-2024**　![Language](https://img.shields.io/badge/language-Jupyter-orange.svg) [![License](https://img.shields.io/badge/license-MIT-blue.svg)](./LICENSE.md) ![AD](https://img.shields.io/badge/深度学习-感知算法-pink.svg)


本项目涵盖了**大模型(LLMs)专题**、**计算机视觉与感知算法专题**、**深度学习基础与框架专题**、**自动驾驶、智慧医疗等行业垂域专题**、**手撕项目代码专题**、**优异开源资源推荐专题**共计6大专题模块。我们将持续整理汇总最新的面试题并详细解析这些题目，希望能成为大家斩获offer路上一份有效的辅助资料。

2024算法面试题目持续更新，具体请 follow [2024年深度学习算法与大模型面试指南](https://github.com/315386775/DeepLearing-Interview-Awesome-2024)，喜欢本项目的请右上角点个star，同时也欢迎大家一起共创该项目。

该项目是持续更新：

- 本文录入面试题的原则：各大厂公司高频算法面试题，大模型领域的面试题，面向业务场景的面试题；
- 目前录入列表的题目，存在部分没有答案解析的题目，或者解析内容不全的题目，我们会尽快补上所有解析；
- 目前录入列表的顺序，没有先后、频次、难度、细类别等维度信息，后续会再给予更多维度更详细的分类；

<b><summary>🏆大模型(LLMs)专题</summary></b>

| [**01. 大模型常用微调方法LORA和Ptuning的原理**](LLMs/Reference.md) |
| :------------------------------------------- |
| [**02. 介绍一下stable diffusion的原理**](LLMs/Reference.md)           | 
| [**03. 为何现在的大模型大部分是Decoder only结构**](LLMs/Reference.md)           | 
| [**04. 如何缓解 LLMs 复读机问题**](LLMs/Reference.md)           | 
| [**05. 为什么transformer中使用LayerNorm而不是BatchNorm**](LLMs/Reference.md)           | 
| [**06. Transformer为何使用多头注意力机制**](LLMs/Reference.md)           | 
| [**07. 监督微调SFT后LLM表现下降的原因**](LLMs/Reference.md)           | 
| [**08. 微调阶段样本量规模增大导致的OOM错误**](LLMs/Reference.md)           | 
| [**09. Attention计算复杂度以及如何改进**](LLMs/Reference.md)           | 
| [**10. BERT用于分类任务的优点，后续改进工作有哪些？**](LLMs/Reference.md)           | 
| [**11. SAM分割一切网络中的Promot类型以及如何输入进网络**](LLMs/Reference.md)           | 
| [**12. Transformer的层融合是如何做到的，其中Residue Network与Layer Norm如何算子融合**](LLMs/Reference.md)           | 
| [**13. 简单介绍下Transformer算法**](LLMs/Reference.md)           | 


<b><summary>🍳计算机视觉与感知算法专题</summary></b>

| [**01. 人脸识别任务中，ArcFace为什么比CosFace效果好**](VisionPerception/Reference.md) |
| :------------------------------------------- |
| [**02. FCOS如何解决重叠样本，以及centerness的作用**](VisionPerception/Reference.md)           | 
| [**03. Centernet为什么可以去除NMS，以及正负样本的定义**](VisionPerception/Reference.md)           | 
| [**04. 介绍CBAM注意力**](VisionPerception/Reference.md)           | 
| [**05. 介绍mixup及其变体**](VisionPerception/Reference.md)           | 
| [**06. Yolov5的正负样本定义**](VisionPerception/Reference.md)           | 
| [**07. Yolov5的一些相关细节**](VisionPerception/Reference.md)           | 
| [**07. Yolov5与Yolov4相比neck部分有什么不同**](VisionPerception/Reference.md)           | 
| [**08. Yolov7的正负样本定义**](VisionPerception/Reference.md)           | 
| [**09. Yolov8的正负样本定义**](VisionPerception/Reference.md)           | 
| [**10. Yolov5的Foucs层和Passthrough层有什么区别**](VisionPerception/Reference.md)           | 
| [**11. DETR的检测算法的创新点**](VisionPerception/Reference.md)           | 
| [**12. CLIP的核心创新点**](VisionPerception/Reference.md)           | 
| [**13. 目标检测中旋转框IOU的计算方式**](VisionPerception/Reference.md)           | 
| [**14. 局部注意力如何实现**](VisionPerception/Reference.md)           | 
| [**15. 视觉任务中的长尾问题的常见解决方案**](VisionPerception/Reference.md)           | 
| [**16. Yolov5中的objectness的作用**](VisionPerception/Reference.md)           | 
| [**17. 匈牙利匹配方法介绍**](VisionPerception/Reference.md)           | 
| [**18. Focal loss的参数如何调，以及存在什么问题**](VisionPerception/Reference.md)           | 
| [**19. 训练一个二分类任务，其中数据有80%的标注正确，20%标注失败**](VisionPerception/Reference.md) |      |  
| [**20. 目标检测设置很多不同的anchor，能否改善小目标及非正常尺寸目标的性能，除计算速度外还存在什么问题**](VisionPerception/Reference.md) |      |  
| [**21. Anchor-free的target assign怎么解决多个目标中心点位置比较靠近的问题**](VisionPerception/Reference.md) |      |  
| [**22. 如果在分类任务中几个类别有重叠（类间差异小）怎么办，如何设计网络结构**](VisionPerception/Reference.md) |      |  


<b><summary>⏰深度学习基础与框架专题</summary></b>

| [**01. 卷积和BN如何融合提升推理速度**](DeepLearning/Reference.md) |
| :------------------------------------------- |
| [**02. 多卡BN如何处理**](DeepLearning/Reference.md) | 
| [**03. TensorRT为什么能让模型跑更快**](DeepLearning/Reference.md) | 
| [**04. 损失函数的应用-合页损失**](DeepLearning/Reference.md) | 
| [**05. Pytorch DataLoader的主要参数有哪些**](DeepLearning/Reference.md) | 
| [**06. 神经网络引入注意力机制后效果降低的原因**](DeepLearning/Reference.md) |  
| [**07. 为什么交叉熵可以作为损失函数**](DeepLearning/Reference.md) |  
| [**08. 优化算法之异同 SGD/AdaGrad/Adam**](DeepLearning/Reference.md) |  
| [**09. 有哪些权重初始化的方法**](DeepLearning/Reference.md) |  
| [**10. MMengine的一些特性**](DeepLearning/Reference.md) |  
| [**11. Modules的一些属性问题**](DeepLearning/Reference.md) |  
| [**12. 激活函数的对比与优缺点**](DeepLearning/Reference.md) |  
| [**13. Transformer/CNN/RNN的时间复杂度对比**](DeepLearning/Reference.md) |  
| [**14. 深度可分离卷积**](DeepLearning/Reference.md) |  
| [**15. CNN和MLP的区别**](DeepLearning/Reference.md) |  
| [**16. MMCV中Hook机制简介及创建一个新的Hook**](DeepLearning/Reference.md) | 
| [**17. 深度学习训练中如何区分错误样本和难例样本**](DeepLearning/Reference.md)           |  
| [**18. PyTorch 节省显存的常用策略**](DeepLearning/Reference.md)           |  
| [**19. 深度学习模型训练时的Warmup预热学习率作用**](DeepLearning/Reference.md)           |  
| [**20. MMdetection中添加一个自定义的backbone网络，需要改哪些代码**](DeepLearning/Reference.md)           |  
| [**21. PyTorch中的 ModuleList 和 Sequential的区别和使用场景**](DeepLearning/Reference.md)           |  


<b><summary>🛺自动驾驶、智慧医疗等行业垂域专题</summary></b>

| [**01. 相机内外参数**](IndustryAlgorithm/Reference.md) |
| :------------------------------------------- |
| [**02. 坐标系的变换**](IndustryAlgorithm/Reference.md) |
| [**03. 放射变换与逆投影变换分别是什么**](IndustryAlgorithm/Reference.md) |      |  
| [**04. 卡尔曼滤波Q和R怎么调**](IndustryAlgorithm/Reference.md) |      |  
| [**05. 如何理解BEV空间及生成BEV特征**](IndustryAlgorithm/Reference.md) |      |  
| [**06. 如何在标注存在错误的数据上训练模型**](IndustryAlgorithm/Reference.md) |      |  
| [**07. 视频与图像中的目标检测具体有什么区别**](IndustryAlgorithm/Reference.md) |      |  
| [**08. 栏杆检测为什么不用网络学习**](IndustryAlgorithm/Reference.md) |      |  
| [**09. 卡尔曼滤波怎么用同一个filter同时适配车辆横穿的场景**](IndustryAlgorithm/Reference.md) |      |  
| [**10. BEV特征怎么进行数据增强**](IndustryAlgorithm/Reference.md) |      |  



<b><summary>🏳‍🌈手撕项目代码专题</summary></b>

| [**01. Pytorch实现注意力机制、多头注意力**](CodeAnything/Reference.md) |
| :------------------------------------------- |
| [**02. Numpy广播机制实现矩阵间L2距离的计算**](CodeAnything/Reference.md) | 
| [**03. Conv2D卷积的Python和C++实现**](CodeAnything/Reference.md) |      |  
| [**04. Numpy实现bbox_iou的计算**](CodeAnything/Reference.md) |      |  
| [**05. Numpy实现Focalloss**](CodeAnything/Reference.md) |      |  
| [**06. Python实现非极大值抑制nms、softnms**](CodeAnything/Reference.md) |      |  
| [**07. Python实现BN批量归一化**](CodeAnything/Reference.md) |      |  
| [**08. Pytorch手写Conv+Bn+Relu，及如何合并**](CodeAnything/Reference.md) |      |  
| [**09. 描述图像resize的过程并实现**](CodeAnything/Reference.md) |      |  
| [**10. PyTorch卷积与BatchNorm的融合**](CodeAnything/Reference.md) |      |  
| [**11. 分割网络损失函数Dice Loss代码实现**](CodeAnything/Reference.md) |      |  


<b><summary>🚩优异开源资源推荐专题</summary></b>

| [**01. 多个优异的数据结构与算法项目推荐**](AwesomeProjects/Reference.md) |
| :------------------------------------------- |
| [**02. 大模型岗位面试总结：共24家，9个offer**](AwesomeProjects/Reference.md)           |  
| [**03. 视觉检测分割一切源码及在线Demo**](AwesomeProjects/Reference.md)           |  
| [**04. 动手学深度学习Pytorch**](AwesomeProjects/Reference.md)           |  




