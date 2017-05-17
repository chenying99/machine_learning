# 中文文档

    * Created on 2017年05月15日
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
        python classification/rnn/lstm/train.py
    5、预测
        注: 先在eval.py中设置checkpoint_dir的路径
        python classification/rnn/lstm/predict.py

