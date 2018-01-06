# Video Highlight Detection

Video Highlight Detection 与 Video Action Detection类似，可以借鉴Video Action Detection的一些思路。Video Action Detection主要是16,17两年的工作，CVPR上有很多相关的工作。其中一种做法是把此任务分为Temporal Action Proposal 和 Temporal Action Classification

## Temporal Action Proposal 

Temporal Action Proposal 和 Object Detection里面Proposal类似。因为我们这次的任务是video highlight检测，和Proposal的任务类似，感觉这个方法可以试试。

1. Paper: **TURN TAP: Temporal Unit Regression Network for Temporal Action Proposals**

   - 这篇文章思路和Faster-RCNN类似,pipeline比较简单


   - Temporal Unit Regression Network (TURN)

     - jointly predicts action proposals and refines the temporal boundaries by temporal coordinate regression
     - unit feature reuse

   - TURN details

     1. decomposed to short video units (e.g. 16 or 32 frames)
     2. For each unit, extract unit-level visual features using C3D and 2-stream CNN, features from  a set of contiguous units are called a clip.
     3. Use multiple temporal scales to create clip features pyramid and concatenate them. each clip is treated as a proposal candidate.
     4. To better estimate the boundary, TURN outputs two regression offsets for the starting time and end time.
     5. Use Non-maximum suppression to remove redundant proposals. 
     6. state-of-art generalization performance

   - Network

     ![TURN](/home/lzk/Documents/paper/paddlepaddle-keci/TURN.png)

   - 主要步骤

     1. Video Unit Processing

        将视频划分为互不重叠的unit，一个unit包括8 frames（这个可以调整），然后使用3D卷积去提取unit-level的特征

     2. Clip Pyramid Modeling

        把每个unit当成一个锚点，前后多个unit构造clip的特征，这个特征由clip内部特征（internal features）和上下文特征组成（context features）	

        $$ f_{c} = P(\{u_{j}\}_{s_{u} - n_{ct_{x}}}^{s_{u}}) || P(\{u_{j}\}_{s_{u}}^{e_{u}}) || P(\{u_{j}\}_{e_{u}}^{e_{u}+n_{c}t_{x}})\\$$

        P代表mean pooling，||代表concatenate， 

        使用多种时间尺度构建每个clip

     3. Unit-level Temporal Coordinate Regression

        这一步算是这篇文章的亮点，计算了回归偏移量。人可以推测一个action结束的时间，这里让网络去学习这个特征。unit regression model有两个输出层，一个是衡量是不是action的confidence score，另一个是回归偏移量，这里用的都是全连接

        $$o_{s} = s_{clip} - s_{gt}, o_{e} = e_{clip} - e_{gt}\\$$ 

        $$s_{clip}$$ 与$$e_{clip}$$ 是输入clip的开始unit和结束unit，$$s_{gt}$$ 与$$e_{gt}$$ 是ground truth。 

     4. Loss Function

        这里，使用TURN给每个clip打上一个二分类的标签（是否是action）

        正样本定义为：1）与其中一个Ground Truth的tIoU最大的样本 2）与任何Ground Truth的tIoU大于0.5的样本

        负样本定义为：和所有GT的tIoU都为0.0的样本（这里不太明白，那其余的呢，标为什么）

        total loss = classification loss +coefficient × regression loss

        $$L = L_{cls} + \lambda L_{reg} \\ $$

        $$L_{reg} = \frac{1}{N_{pos}}\sum_{i=1}^{N}l_{i}^{*}|(o_{s,i} - o_{s,i}^{*}) + (o_{e,i} - o_{e,i}^{*})|\\$$ 

        $\lambda$ 是超参数， $l_{i}^{*}$ 是label，$N_{pos}$ 是正样本的数量。

     5. https://github.com/jiyanggao/TURN-TAP 作者开源的（代码风格很差...）

## Temporal Action classification

1. Paper: **Action Classification and Highlighting in Videos**

   ​



