# 中文文档

    * Created on 2017年05月15日
    * author: MJ
    * project: text_cnn

本项目是在原有的代码基础上进行适当的修改，并添加一些中文注释，方便广大读者理解TextCNN模型的使用


## 参考资料
    原paper：
         - [Convolutional Neural Networks for Sentence Classification](http://arxiv.org/abs/1408.5882)
         - [A Sensitivity Analysis of (and Practitioners' Guide to) Convolutional Neural Networks for Sentence Classification](http://arxiv.org/abs/1510.03820)
    源代码：
        - https://github.com/dennybritz/cnn-text-classification-tf
    原博客：
        - http://www.wildml.com/2015/12/implementing-a-cnn-for-text-classification-in-tensorflow/

## 1. 数据和预处理
    数据集：电影评论数据(Movie Review data from Rotten Tomatoes),包含5331个积极评论和5331个消极评论，同时包含一个20k的词表
    注意：数据集过小容易过拟合，可以使用10%的训练数据进行交叉验证
    步骤：
        (1) 加载两类数据
        (2) 文本数据清洗
        (3) 把每个句子填充到最大的句子长度，填充字符是<PAD>，使得每个句子都包含59个单词。相同的长度有利于进行高效的批处理
        (4) 根据所有单词的词表，建立一个索引，用一个整数代表一个词，则每个句子由一个整数向量表示
## 2. 模型
    2.1 步骤
        (1) 第一层把词嵌入到低维向量；
        (2) 第二层用多个不同大小的filter进行卷积；
        (3) 第三层用max-pool把第二层多个filter的结果转换成一个长的特征向量并加入dropout正规化；
        (4) 第四层用softmax进行分类。

    2.2 简化模型，方便理解
        (1) 不使用预训练的word2vec的词向量，而是学习如何嵌入
        (2) 不对权重向量强制执行L2正规化
        (3) 原paper使用静态词向量和非静态词向量两个同道作为输入，这里只使用一种同道作为输入
## 3. 实现
    3.1 TextCNN类，config参数如下：
        filter_sizes：卷积核的高度
        num_filters：每种不同大小的卷积核的个数，这里每种有3个
        embedding_dim ：嵌入的维度
        num_classes：输出的类型个数，这里是积极和消极两类
        dropout_keep_prob：保留一个神经元的概率，这个概率只在训练的时候用到
        initial_learning_rate：初始的学习率
        min_learning_rate：学习率最小值
        decay_rate：学习率衰减率
        decay_step：学习率衰减步长
        batch_size：每批读入样本的数量
        num_step：句子长度，把每个句子统一填充到最大长度
        l2_reg_lambda：L2 正则项的参数lambda
        vocabulary_size：词典长度，需要在嵌入层定义
    3.2 输入占位符（定义我们要传给网络的数据）
        如输入占位符，输出占位符和dropout占位符
        tf.placeholder创建一个占位符，在训练和测试时才会传入相应的数据
            第一个参数是数据类型;
            第二个参数是tensor的格式，none表示是任何大小;
            第三个参数是名称;
        dropout_keep_prob是保留一个神经元的概率，这个概率只在训练的时候用到
    3.3 第一层（嵌入层）
        tf.device("/cpu:0"),表示使用cpu进行操作，因为tensorflow当gpu可用时默认使用gpu，但是embedding不支持gpu实现，所以使用CPU操作
        tf.name_scope("embedding"),把所有操作加到命名为embedding的顶层节点，用于可视化网络视图
        W是在训练时得到的嵌入矩阵，通过随机均匀分布进行初始化
        tf.nn.embedding_lookup 是真正的embedding操作，结果是一个三维的tensor，[None, sequence_length, embedding_size]
        因为卷积操作conv2d需要4个维度的tensor所以需要给embedding结果增加一个维度，得到[None, sequence_length, embedding_size, 1]
    3.4 卷积和max-pooling
        对不同大小的filter建立不同的卷积层，W是卷积的输入矩阵，h是使用relu进行卷积的结果。
        “VALID”表示使用narrow卷积，得到的结果大小为[1, sequence_length - filter_size + 1, 1, 1]
        为了更容易理解，需要计算输入输出的大小
    3.5 Dropout层
        dropout是正规化卷积神经网络最流行的方法，即随机禁用一些神经元
    3.6 分数和预测
        用max-pooling得到的向量x作为输入，与随机产生的W权重矩阵进行计算得到分数，选择分数高的作为预测类型结果
    3.7 交叉熵损失和正确率
    3.8 网络可视化
    3.9 训练过程
        (1) Session是执行graph操作（表示计算任务）的上下文环境，包含变量和序列的状态。
            每个session执行一个graph。tensorflow包含了默认session，也可以自定义session然后通过session.as_default() 设置为默认视图
        (2) graph包含操作和tensors（表示数据），可以在程序中建立多个图，但是通常只需一个图。
            同一个图可以在多个session中使用，但是不能多个图在一个session中使用。
        (3) allow_soft_placement可以在不存在预设运行设备时可以在其他设备运行
            例如设置在gpu上运行的操作，当没有gpu时allow_soft_placement使得可以在cpu操作
        (4) log_device_placement用于设备的log，方便debugging
        (5) FLAGS是程序的命令行输入
    3.10 CNN初始化和最小化loss
        按照TextCNN的参数进行初始化
        tensorflow提供了几种自带的优化器，我们使用Adam优化器求loss的最小值
        train_op就是训练步骤，每次更新我们的参数，global_step用于记录训练的次数，在tensorflow中自增
    3.11 summaries汇总
        tensorflow提供了各方面的汇总信息，方便跟踪和可视化训练和预测的过程。summaries是一个序列化的对象，通过tf.summary.FileWriter写入到光盘
    3.12 checkpointing检查点
        用于保存训练参数，方便选择最优的参数，使用tf.train.saver()进行保存
    3.13 变量初始化
        sess.run(tf.global_variables_initializer())，用于初始化所有我们定义的变量，也可以对特定的变量手动调用初始化，如预训练好的词向量
    3.14 定义单一的训练步骤
        feed_dict中包含了我们在网络中定义的占位符的数据，必须要对所有的占位符进行赋值，否则会报错
        train_op不返回结果，只是更新网络的参数
    3.15 训练循环
        遍历数据并对每次遍历数据调用train_step函数，并定期打印模型评价和检查点
    3.16 用tensorboard进行结果可视化
        tensorboard --logdir=path/to/log-directory
    3.17 本实验的几个问题
        训练的指标不是平滑的，原因是我们每个批处理的数据过少
        训练集正确率过高，测试集正确率过低，过拟合。
        避免过拟合：更多的数据；更强的正规化；更少的模型参数。
        例如对最后一层的权重进行L2惩罚，使得正确率提升到76%，接近原始paper

## 运行
    1、先切换到项目所在路径
        cd /Users/MJ/machine_learning
    2、激活虚拟环境
        source ../ENV_machine_learning/bin/activate
    3、训练
        python classification/cnn/text_cnn/train.py
        输出示例:
        Train, epoch 20, step 3210, lr 0.000512, loss 0.011897, acc 1, step-time 0.481014, examples/sec 133.052
        Train, epoch 20, step 3220, lr 0.000512, loss 0.00960575, acc 1, step-time 0.231911, examples/sec 275.968
        Train, epoch 20, step 3230, lr 0.000512, loss 0.00326563, acc 1, step-time 0.247206, examples/sec 258.894
        Train, epoch 20, step 3240, lr 0.000512, loss 0.0114532, acc 1, step-time 0.279439, examples/sec 229.03
        Train, epoch 20, step 3250, lr 0.000512, loss 0.00550958, acc 1, step-time 0.253991, examples/sec 251.978
        Train, epoch 20, step 3260, lr 0.000512, loss 0.00892533, acc 1, step-time 0.296604, examples/sec 215.776
        Train, epoch 20, step 3270, lr 0.000512, loss 0.0237666, acc 0.984375, step-time 0.296659, examples/sec 215.736
        Train, epoch 20, step 3280, lr 0.000512, loss 0.0132664, acc 1, step-time 0.327229, examples/sec 195.582
        Train, epoch 20, step 3290, lr 0.000512, loss 0.00794089, acc 1, step-time 0.307061, examples/sec 208.428
        Train, epoch 20, step 3300, lr 0.000512, loss 0.0155414, acc 1, step-time 0.303007, examples/sec 211.216
        Valid, step 3300, lr 0.000512, loss 0.747256, acc 0.767
    4、评估
        注: 先在eval.py中设置checkpoint_dir的路径
        python classification/cnn/text_cnn/eval.py
        输出示例:
        Total number of test examples: 10662
        Accuracy: 0.977772
        Saving evaluation to /Users/MJ/machine_learning/classification/cnn/text_cnn/data/model/runs/1494832207/checkpoints/../prediction.csv

    5、预测一条文本
        注: 先在predict.py中设置checkpoint_dir的路径
        python classification/cnn/text_cnn/predict.py
