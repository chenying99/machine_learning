# 中文文档

    * Created on 2017年05月16日
    * author: MJ
    * project: lstm


## 参考资料
    原paper：
        - [Long Short-term Memory](https://www.researchgate.net/profile/Sepp_Hochreiter/publication/13853244_Long_Short-term_Memory/links/5700e75608aea6b7746a0624/Long-Short-term-Memory.pdf)
        - [Bidirectional Recurrent Neural Networks](http://www.di.ufpe.br/~fnj/RNA/bibliografia/BRNN.pdf)

##  数据和预处理
    数据集：搜狗新闻分类数据,包含9个类别,每个类别包含1990个样本
    注意：数据集过小容易过拟合，可以使用10%的训练数据进行交叉验证
    步骤：
        (1) 加载数据
        (2) 分词、停词处理
        (3) 训练的word2vec模型
        (4) 根据训练好的word2vec模型, 每个词表示为1个词向量, 每个句子表示为2维矩阵

## 运行
    1、先切换到项目所在路径
        cd /Users/MJ/machine_learning
    2、激活虚拟环境
        source ../ENV_machine_learning/bin/activate
    3、训练word2vec
        python word2vec/train.py train_sogou_classification_word2vec
    4、训练模型
        4.1、lstm模型: python classification/rnn/lstm/train.py lstm
            输出示例:
            Valid, step 3700, lr 0.000512, loss 0.727028, acc 0.843
            epoch 99
            Train, epoch 99, step 3710, lr 0.000512, loss 0.157416, acc 0.952, step-time 1.21417, examples/sec 411.805
            Train, epoch 99, step 3720, lr 0.000512, loss 0.117503, acc 0.96, step-time 1.36502, examples/sec 366.294
            Train, epoch 99, step 3730, lr 0.000512, loss 0.200994, acc 0.946, step-time 1.36577, examples/sec 366.093
            epoch 100
            Train, epoch 100, step 3740, lr 0.000512, loss 0.163862, acc 0.948, step-time 0.152858, examples/sec 3271.02
            Train, epoch 100, step 3750, lr 0.000512, loss 0.173707, acc 0.952, step-time 1.36271, examples/sec 366.916
            Train, epoch 100, step 3760, lr 0.000512, loss 0.123907, acc 0.964, step-time 1.36246, examples/sec 366.983
            Train, epoch 100, step 3770, lr 0.000512, loss 0.136535, acc 0.96, step-time 1.36065, examples/sec 367.471
        4.2、bi_lstm模型: python classification/rnn/lstm/train.py bi_lstm
    5、预测
        注: 先在predict.py中设置checkpoint_dir的路径
        python classification/rnn/lstm/predict.py

