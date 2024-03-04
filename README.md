# **DeepLearning-Interview-Awesome-2024**　![Language](https://img.shields.io/badge/language-Jupyter-orange.svg) [![License](https://img.shields.io/badge/license-MIT-blue.svg)](./LICENSE.md) ![AD](https://img.shields.io/badge/深度学习-感知算法-pink.svg)


本项目涵盖了**大模型(LLMs)专题**、**计算机视觉与感知算法专题**、**深度学习基础与框架专题**、**自动驾驶、智慧医疗等行业垂域专题**、**手撕项目代码专题**、**优异开源资源推荐专题**共计6大专题模块。我们将持续整理汇总最新的面试题并详细解析这些题目，希望能成为大家斩获offer路上一份有效的辅助资料。

2024算法面试题目持续更新，具体请 follow [2024年深度学习算法与大模型面试指南](https://github.com/315386775/DeepLearing-Interview-Awesome-2024)，喜欢本项目的请右上角点个star，同时也欢迎大家一起共创该项目。

该项目持续更新：

- 本文录入题目的原则：**高新深**，其中高是指-各大厂公司近年高频算法面试题，新是指-题目要新紧跟学术和工业界的发展，比如录入了大量大模型领域的面试题，深是指-题目要有一定的内容与深度，可以引人思考，比如面向业务场景改进的面试题；
- 目前录入列表的题目，存在部分没有答案解析的题目，或者解析内容不全的题目，我们会尽快补上所有解析；
- 目前录入列表的顺序，没有先后、频次、难度、细类别等维度信息，后续会再给予更多维度更详细的分类（TODO:题目顺序归类中，答案顺序未整理）；

<b><summary>🏆大模型(LLMs)专题</summary></b>

- 大语言模型

| [**01. 模型微调：大模型常用微调方法LORA和Ptuning的原理**](LLMs/Reference.md) |
| :------------------------------------------- |
| [**30. 模型微调：Instruction Tuning与Prompt tuning方法的区别？**](LLMs/Reference.md)           | 
| [**07. 模型微调：监督微调SFT后LLM表现下降的原因**](LLMs/Reference.md)           | 
| [**18. 模型微调：大模型微调的LORA怎么训练？**](LLMs/Reference.md)           | 
| [**19. 模型微调：LORA的矩阵怎么初始化？为什么要初始化为全0？**](LLMs/Reference.md)           | 
| [**33. 模型微调：进行SFT操作的时候，基座模型选用Chat还是Base?**](LLMs/Reference.md)           |
| [**03. 模型结构：为何现在的大模型大部分是Decoder only结构**](LLMs/Reference.md)           | 
| [**15. 模型结构：你能否概括介绍一下 ChatGPT 的训练过程？**](LLMs/Reference.md)           | 
| [**16. 模型结构：在大型语言模型 (llms) 上下文中的标记是什么？**](LLMs/Reference.md)           | 
| [**40. 模型结构：GPT3、LLAMA的Layer Normalization 的区别是什么？**](LLMs/Reference.md)           | 
| [**04. 模型优化：如何缓解 LLMs 复读机问题**](LLMs/Reference.md)           | 
| [**14. 模型优化：在大型语言模型 (llms) 中减少幻觉的策略有哪些？**](LLMs/Reference.md)           | 
| [**29. 模型优化：如何提升大语言模型的Prompt泛化性？**](LLMs/Reference.md)           | 
| [**34. 模型优化：开源大模型进行预训练的过程中会加入书籍、论文等数据，这部分数据如何组织与处理?**](LLMs/Reference.md)           | 
| [**38. 模型优化：如何解决chatglm微调的灾难性遗忘问题？**](LLMs/Reference.md)           | 
| [**10. BERT用于分类任务的优点，后续改进工作有哪些？**](LLMs/Reference.md)           | 
| [**23. BERT的预训练任务有什么？为什么引入下一个句子预测任务？**](LLMs/Reference.md)           | 
| [**37. BERT的预训练过程中是否使用了位置编码和注意力机制？**](LLMs/Reference.md)           | 

- 视觉模型

| [**01. Stable Diffusion里是如何用文本来控制生成的？**](LLMs/Reference.md) |
| :------------------------------------------- |
| [**21. Stable Diffusion相比Diffusion主要解决的问题是什么？**](LLMs/Reference.md)           | 
| [**22. Stable Diffusion每一轮训练样本选择一个随机时间步长？**](LLMs/Reference.md)           | 
| [**39. Stable Diffusion的训练过程和预测过程是什么样的？**](LLMs/Reference.md)           | 
| [**11. 基座模型：SAM分割一切网络中的Promot类型以及如何输入进网络**](LLMs/Reference.md)           | 
| [**26. 基座模型：训练通用目标检测器常会使用多源图像进行训练，如何处理新类别歧视？**](LLMs/Reference.md)           | 

- 通用问题

| [**01. 为什么Transformer中使用LayerNorm而不是BatchNorm？**](LLMs/Reference.md) |
| :------------------------------------------- |
| [**06. Transformer为何使用多头注意力机制**](LLMs/Reference.md)           | 
| [**32. Transformer中的Attention计算复杂度以及如何改进？**](LLMs/Reference.md)           | 
| [**12. Transformer的层融合是如何做到的，其中Residue Network与Layer Norm如何算子融合**](LLMs/Reference.md)           | 
| [**41. MHA多头注意力和MQA多查询注意力的区别？**](LLMs/Reference.md)           | 
| [**17. Adaptive Softmax在大型语言模型中有何用处？**](LLMs/Reference.md)           | 
| [**31. 知识蒸馏是将复杂模型的知识转移到简单模型的方法，针对知识蒸馏有哪些改进点？**](LLMs/Reference.md)           | 
| [**42. 推理优化技术 Flash Attention 的作用是什么？**](LLMs/Reference.md)           | 
| [**43. ZeRO，零冗余优化器的三个阶段？**](LLMs/Reference.md)           | 

- 多模态模型/强化学习/AGI等

| [**01. 举例说明强化学习如何发挥作用？**](LLMs/Reference.md) |
| :------------------------------------------- |
| [**28. 如何理解强化学习中的奖励最大化？**](LLMs/Reference.md)           | 
| [**24. 领域数据训练后，通用能力往往会有所下降，如何缓解模型遗忘通用能力？**](LLMs/Reference.md)           | 
| [**25. 在大型语言模型 (llms)中数据模态的对齐如何处理？**](LLMs/Reference.md)           | 
| [**35. 你能提供一些大型语言模型中对齐问题的示例吗？**](LLMs/Reference.md)           | 


<b><summary>🍳计算机视觉与感知算法专题</summary></b>

- 常见问题

| [**01. 大卷积核：在CNN网络中更大的核是否可以取得更高的精度？**](VisionPerception/Reference.md) |
| :------------------------------------------- |
| [**02. 优化算法：匈牙利匹配方法可用于正负样本定义等问题中，介绍其实现原理**](VisionPerception/Reference.md)           | 
| [**03. 损失函数：Focal loss的参数如何调，以及存在什么问题**](VisionPerception/Reference.md)           | 
| [**04. 模型轻量化：举例一些从参数量、浮点运算量、模型推理时延进行优化，具有代表性的轻量化模型？**](VisionPerception/Reference.md) |
| [**05. 图像处理：ORB特征提取的缺陷及如何进行改进**](VisionPerception/Reference.md) |
| [**06. 通用模块：FPN的特征融合为什么是相加操作呢？**](VisionPerception/Reference.md) | 
| [**07. 通用模块：如何理解concat和add这两种常见的feature map特征融合方式？**](VisionPerception/Reference.md) | 
| [**08. 通用模块：Transformer的注意力机制常用softmax函数，可以使用sigmoid代替吗？**](VisionPerception/Reference.md) | 

- 目标分类

| [**01. 损失函数：人脸识别任务中，ArcFace为什么比CosFace效果好**](VisionPerception/Reference.md) |
| :------------------------------------------- |
| [**02. 通用模块：介绍CBAM注意力**](VisionPerception/Reference.md)           | 
| [**03. 通用模块：局部注意力如何实现**](VisionPerception/Reference.md)           | 
| [**04. 数据增强：介绍mixup及其变体**](VisionPerception/Reference.md)           | 
| [**05. 场景问题：视觉任务中的长尾问题的常见解决方案**](VisionPerception/Reference.md)           | 
| [**06. 场景问题：如果在分类任务中几个类别有重叠（类间差异小）怎么办，如何设计网络结构**](VisionPerception/Reference.md) |
| [**07. 场景问题：在A场景下进行目标的标记及训练，如何在B场景下取得好的效果？**](VisionPerception/Reference.md) | 
| [**08. 场景问题：如何更好的训练一个二分类任务，其中数据有80%的标注正确，20%标注失败**](VisionPerception/Reference.md) |      |  
| [**09. 基座模型：CLIP的核心创新点简介，其如何处理文本输入**](VisionPerception/Reference.md) |      |  
| [**10. 基座模型：ViT、DEIT是如何处理变长序列输入的？**](VisionPerception/Reference.md) | 
| [**11. 基座模型：VIT中对输入图像的处理是如何将patch变化为token的？**](VisionPerception/Reference.md) |

- 目标检测

| [**01. 样本匹配策略：FCOS训练阶段如何解决重叠样本造成的GT不一致问题**](VisionPerception/Reference.md) |
| :------------------------------------------- |
| [**02. 样本匹配策略：Centernet为什么可以去除NMS，以及正负样本的定义**](VisionPerception/Reference.md)           |
| [**03. 样本匹配策略：Yolov5的正负样本定义，一个目标是否会被分配到不同的FPN层中**](VisionPerception/Reference.md)           | 
| [**04. 样本匹配策略：Yolov7的正负样本定义**](VisionPerception/Reference.md)           | 
| [**05. 样本匹配策略：Yolov8的正负样本定义**](VisionPerception/Reference.md)           |  
| [**06. 样本匹配策略：Yolov9的正负样本定义**](VisionPerception/Reference.md)           |  
| [**07. 样本匹配策略：Yolov1的正负样本定义**](VisionPerception/Reference.md)           |  
| [**08. 样本匹配策略：DETR用二分图匹配实现label assignment，简述其过程**](VisionPerception/Reference.md)           |  
| [**09. 样本匹配策略：Anchor-free的target assign怎么解决多个目标中心点位置比较靠近的问题**](VisionPerception/Reference.md)           | 
| [**10. 样本匹配策略：Anchor-Based检测器在正负样本标签分配阶段，如何去除对anchor的依赖？**](VisionPerception/Reference.md) | 
| [**11. 样本匹配策略：目标检测如何选取正负样本将会极大的影响最后的检测效果，举例ATSS如何处理的？**](VisionPerception/Reference.md) |
| [**12. 损失函数优化：FCOS的损失函数中centerness的作用**](VisionPerception/Reference.md) |
| [**13. 损失函数优化：有哪些可以解决目标检测中正负样本不平衡问题的方法**](VisionPerception/Reference.md) | 
| [**14. 细节问题：Yolov5与Yolov4相比neck部分有什么不同**](VisionPerception/Reference.md)           | 
| [**15. 细节问题：Yolov5的Foucs层和Passthrough层有什么区别**](VisionPerception/Reference.md)           | 
| [**16. 细节问题：Yolov5中objectness的作用，最后输出的概率分数如何得到**](VisionPerception/Reference.md)           | 
| [**17. 模型问题：DETR的检测算法的创新点介绍**](VisionPerception/Reference.md)           | 
| [**18. 解码问题：解释YOLOv5模型输出(1, 25200, 85)的含义，及解码过程？**](VisionPerception/Reference.md) | 
| [**19. 解码问题：解释Centernet模型输出offset/scale/heatmap三个头的含义，及解码过程？**](VisionPerception/Reference.md) |
| [**20. 场景问题：目标检测中旋转框IOU的计算方式**](VisionPerception/Reference.md)           | 
| [**21. 场景问题：如何修改Yolov5目标检测，从而实现旋转目标检测？**](VisionPerception/Reference.md) | 
| [**22. 场景问题：在目标Crowded的场景下，经常在两个真正目标中间会出现误检的原因?**](VisionPerception/Reference.md) |
| [**23. 场景问题：通过设置更多的先验anchor能否改善小目标及非正常尺寸目标的性能，除计算速度外还存在什么问题**](VisionPerception/Reference.md) |

- 目标分割

| [**01. 模型问题：在Unet网络结构中，四次降采样对于分割网络到底是不是必须的？**](VisionPerception/Reference.md) |
| :------------------------------------------- |
| [**02. 模型问题：为什么UNet++可以被剪枝，怎么去决定剪多少？**](VisionPerception/Reference.md)           | 
| [**03. 模型问题：分割一切网络SAM如何处理目标的分割掩码输出？**](VisionPerception/Reference.md) |
| [**04. 模型问题：SAM在本地的模型推理效果明显差于线上web版本，有什么方式可以优化其效果？**](VisionPerception/Reference.md) |
| [**05. 基座模型：VIT直接用于分割检测等预测密集型的任务上存在什么问题？**](VisionPerception/Reference.md) |


- 对抗网络/视频理解/图像增强/深度估计等

| [**01. 对抗网络：GAN中的模式坍缩的识别和解决？**](VisionPerception/Reference.md) |
| :------------------------------------------- |
| [**02. 深度估计：简述深度估计任务中常用到的光度重建损失？**](VisionPerception/Reference.md)           |  


<b><summary>⏰深度学习基础与框架专题</summary></b>

- Pytorch常用操作及问题

| [**01. Pytorch 训练时经常会合并多个数据集，ConcatDataset具体做了什么？**](DeepLearning/Reference.md) |
| :------------------------------------------- |
| [**02. Pytorch 的多卡BN如何处理？**](DeepLearning/Reference.md) | 
| [**03. Pytorch DataLoader的主要参数有哪些**](DeepLearning/Reference.md) | 
| [**04. Pytorch 代码中如何尽量避免.to(device)的操作？**](DeepLearning/Reference.md)           |  
| [**05. Pytorch 中nn.Identity()/.chunk/.masked_select/.gather操作的应用场景？**](DeepLearning/Reference.md)           |  
| [**06. PyTorch 节省显存的常用策略**](DeepLearning/Reference.md)           |  
| [**07. PyTorch 的Modules一些属性问题**](DeepLearning/Reference.md) |  
| [**08. PyTorch 中的 ModuleList 和 Sequential的区别和使用场景**](DeepLearning/Reference.md)           | 

- 那些常用的训练框架

| [**01. TensorRT 为什么能让模型跑的更快**](DeepLearning/Reference.md) |
| :------------------------------------------- |
| [**02. MMengine 的一些特性，其基础配置包含哪些内容**](DeepLearning/Reference.md) |  
| [**03. MMdetect 中添加一个自定义的backbone网络，需要改哪些代码**](DeepLearning/Reference.md)           |  
| [**04. MMCV 中Hook机制简介及创建一个新的Hook**](DeepLearning/Reference.md) | 

- 深度学习常见问题

| [**01. 算子问题：卷积和BN如何融合提升推理速度**](DeepLearning/Reference.md) |
| :------------------------------------------- |
| [**02. 算子问题：神经网络引入注意力机制后效果降低的原因**](DeepLearning/Reference.md) |  
| [**03. 算子问题：激活函数的对比与优缺点**](DeepLearning/Reference.md) |  
| [**04. 算子问题：Transformer/CNN/RNN的时间复杂度对比**](DeepLearning/Reference.md) |  
| [**05. 算子问题：深度可分离卷积**](DeepLearning/Reference.md) |  
| [**06. 算子问题：CNN和MLP的区别**](DeepLearning/Reference.md) |  
| [**07. 损失函数：损失函数的应用-合页损失**](DeepLearning/Reference.md) | 
| [**08. 损失函数：为什么交叉熵可以作为损失函数**](DeepLearning/Reference.md) |  
| [**09. 优化算法：优化算法之异同 SGD/AdaGrad/Adam**](DeepLearning/Reference.md) |  
| [**10. 优化算法：有哪些权重初始化的方法**](DeepLearning/Reference.md) |  
| [**11. 优化算法：深度学习中为什么不对 bias 偏置进行正则化？**](DeepLearning/Reference.md)           |  
| [**12. 优化算法：正则化为什么可以增加模型泛化能力**](DeepLearning/Reference.md)           |  
| [**13. 优化算法：为什么Adam常常打不过SGD？症结点与改善方案？**](DeepLearning/Reference.md)           |  
| [**14. 常见问题：深度学习训练中如何区分错误样本和难例样本**](DeepLearning/Reference.md)           |  
| [**15. 常见问题：深度学习模型训练时的Warmup预热学习的作用**](DeepLearning/Reference.md)           |  
| [**16. 常见问题：考虑一个filter[-1 -1 -1; 0 0 0; 1 1 1] 用于卷积，该滤波器将从输入图像中提取哪些边缘**](DeepLearning/Reference.md)           |  
| [**17. 场景问题：深度学习模型中如何融入传统图像处理的特征？直接拼接融合有什么问题？**](DeepLearning/Reference.md)           |  
| [**18. 场景问题：多任务学习中各个任务损失的权重应该如何设计呢？**](DeepLearning/Reference.md)           |  
| [**19. 场景问题：如何处理不平衡的数据集？**](DeepLearning/Reference.md)           |  
| [**20. 场景问题：如何将大模型有效地切割成若干个子模型？如何将切割后的子模型分配到多个节点上进行并行训练？**](DeepLearning/Reference.md)           |  


<b><summary>🛺自动驾驶、智慧医疗等行业垂域专题</summary></b>

- 自动驾驶

| [**01. 相机内参和外参的含义？如果将图像放大两倍，内外参如何变化？**](IndustryAlgorithm/Reference.md) |
| :------------------------------------------- |
| [**02. 从世界坐标系到图像坐标系的变换关系？**](IndustryAlgorithm/Reference.md) |
| [**03. 放射变换与逆投影变换分别是什么**](IndustryAlgorithm/Reference.md) |      |  
| [**04. 卡尔曼滤波Q和R怎么调**](IndustryAlgorithm/Reference.md) |      |  
| [**05. 如何理解BEV空间及生成BEV特征**](IndustryAlgorithm/Reference.md) |      |  
| [**08. 栏杆检测为什么不用网络学习**](IndustryAlgorithm/Reference.md) |      |  
| [**09. 卡尔曼滤波怎么用同一个filter同时适配车辆横穿的场景**](IndustryAlgorithm/Reference.md) |      |  
| [**10. BEV特征怎么进行数据增强**](IndustryAlgorithm/Reference.md) |      |  
| [**11. 辅助驾驶场景中，模型对60米之内的中大目标预测的bbox坐标不稳定，有较大的抖动问题，导致测距不稳定，怎么解决？**](IndustryAlgorithm/Reference.md) |      |  
| [**12. 辅助驾驶场景中，对公交站、房屋顶等特定背景误检，怎么解决？**](IndustryAlgorithm/Reference.md) |      |  
| [**13. 辅助驾驶场景中，大于100m的车辆车型分类出现跳动怎么解决？**](IndustryAlgorithm/Reference.md) |      |  
| [**16. 解释KF中的噪声矩阵含义。运动方程中估计噪声是变大还是变小？修正方程中估计噪声是变大还是变小？**](IndustryAlgorithm/Reference.md)           | 
| [**20. 车道线检测的任务通常采用分割方案，如何将方案降级至检测，甚至是车道线分类？**](IndustryAlgorithm/Reference.md)           | 
| [**21. 车道线检测的任务中如何处理异行线，比如道路交叉口？**](IndustryAlgorithm/Reference.md)           | 
| [**24. 简述BEVformer的Decoder逻辑？**](IndustryAlgorithm/Reference.md)           | 
| [**25. BEVFormer中的Spatial Cross-Attention的步骤？**](IndustryAlgorithm/Reference.md)           | 
| [**26. 车上多个摄像头图像投影到2D平面如何实现？**](IndustryAlgorithm/Reference.md)           | 
| [**27. 假如你的车子有4个激光雷达，你如何设计点云分割算法？**](IndustryAlgorithm/Reference.md)           | 
| [**28. 假如当前需要你把场景里的砖头分割出来，靠点云分割能否正确识别？**](IndustryAlgorithm/Reference.md)           | 
| [**29. 点云中的水雾怎么去除？**](IndustryAlgorithm/Reference.md)           | 
| [**30. 车宽测距和接地点测距分别使用了什么样的先验知识？这些先验如果不成立的时候能有什么手段来放宽限制？**](IndustryAlgorithm/Reference.md)           | 
| [**31. 车辆行驶过程中 Pitch 角度估计的三种方法？**](IndustryAlgorithm/Reference.md)           | 
| [**32. 如何消除一堆3D点云中的角点？**](IndustryAlgorithm/Reference.md)           | 
| [**33. 如何将 3D 世界坐标点转换为 2D 图像坐标？**](IndustryAlgorithm/Reference.md)           | 
| [**34. 单目3D目标检测的预测信息包含哪些，在预测3D框中心偏差时针对截断目标如何处理？**](IndustryAlgorithm/Reference.md)           | 
| [**35. 通过几何关系估计深度过程中，由于高度的误差使得深度的估计不确定性高，如何缓解？**](IndustryAlgorithm/Reference.md)           |

- 智慧医疗

| [**01. 数据标注：医学影像由于标注专业性差异，出现多人标注不一致情况怎么解决？如何用算法的方式减少误差？**](IndustryAlgorithm/Reference.md) |
| :------------------------------------------- |
| [**02. 模型问题：模型中如何添加病史信息来增强最终的分类效果？**](IndustryAlgorithm/Reference.md) |      |  

- 自然语言处理/智慧商业/搜广推

| [**01. 自然语言处理：NLP中给定当前query和历史query以及对应实体，如何对当前query的实体进行建模？**](IndustryAlgorithm/Reference.md) |
| :------------------------------------------- |
| [**02. 机器学习：银行经理收到一个数据集，其中包含数千名申请贷款的申请人的记录。AI算法如何帮助经理了解他可以批准哪些贷款？**](IndustryAlgorithm/Reference.md)           | 

- 场景实战

| [**01. 如何在标注存在错误的数据上训练模型？**](IndustryAlgorithm/Reference.md) |
| :------------------------------------------- |
| [**02. 视频与图像中的目标检测具体有什么区别**](IndustryAlgorithm/Reference.md) |
| [**03. 举出几种光流方法，说明LK光流的建模方式？**](IndustryAlgorithm/Reference.md)           | 
| [**04. 如何在数据量十分有限，但特征数量极多的情况下选出一套合适的特征组合？**](IndustryAlgorithm/Reference.md)           | 
| [**05. SAM的点提示和框提示输入尺寸，框提示是否支持多个框？**](IndustryAlgorithm/Reference.md)           | 
| [**06. 为什么 larger batch size 对对比学习的影响比对监督学习的影响要大？**](IndustryAlgorithm/Reference.md)           | 

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
| [**12. Numpy实现一个函数来计算两个向量之间的余弦相似度**](CodeAnything/Reference.md) |      |  
| [**13. Numpy实现Sigmoid函数**](CodeAnything/Reference.md) |      |  
| [**14. 使用Pytorch搭建一个CNN卷积神经网络**](CodeAnything/Reference.md) |      |  


<b><summary>🚩优异开源资源推荐专题</summary></b>

| [**01. 多个优异的数据结构与算法项目推荐**](AwesomeProjects/Reference.md) |
| :------------------------------------------- |
| [**02. 大模型岗位面试总结：共24家，9个offer**](AwesomeProjects/Reference.md)           |  
| [**03. 视觉检测分割一切源码及在线Demo**](AwesomeProjects/Reference.md)           |  
| [**04. 动手学深度学习Pytorch**](AwesomeProjects/Reference.md)           |  
| [**05. 一种用于保存、搜索、访问、探索和与您喜爱的所有网站、文档和文件聊天的工具**](AwesomeProjects/Reference.md)           |  
| [**06. 收集一些免费的ChatGPT镜像站点**](AwesomeProjects/Reference.md)           |  
| [**07. 关于大型语言模型(LLM)的一切**](AwesomeProjects/Reference.md)           |  
| [**08. 深度学习调优指南中文版**](AwesomeProjects/Reference.md)           |  
| [**09. 多模态大型语言模型的最新论文和数据集集锦**](AwesomeProjects/Reference.md)           |  
| [**10. ChatPaper：ChatGPT来加速科研流程的工具**](AwesomeProjects/Reference.md)           |  
| [**11. 消费级硬件上进行LLaMA的微调**](AwesomeProjects/Reference.md)           |  
| [**12. Stability AI提供的一系列生成模型**](AwesomeProjects/Reference.md)           |  
| [**13. 自监督方式学习强大视觉特征的框架DINOv2**](AwesomeProjects/Reference.md)           |  
| [**14. 快速的场景分割FastSAM**](AwesomeProjects/Reference.md)           |  
| [**15. 语言大模型面试题**](AwesomeProjects/Reference.md)           |  

**欢迎大家一起共创该项目，也可加博主微信探讨交流**

<img src="assert/wechat.png" alt="Alt" style="display: block; margin: 0 auto; width: 500px; height: 430px;">

