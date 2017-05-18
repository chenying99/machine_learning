# 中文文档

    * Created on 2017年05月15日
    * author: MJ
    * project: machine_learning
    * 本人会陆续将自己使用过的机器学习算法示例进行整理,欢迎各位读者挑错指正,如果对你有帮助,请给个star哦!
      如果你对此项目有任何疑问, 本人将在下几个QQ群给予答疑解惑:
      TensorFlow深度学习交流NLP群(299814789)
      TensorFlow深度学习交流CV群(397163918)


## 项目依赖
    * 1、IDE PyCharm Community Edition 2016.1.5
    * 2、安装 python 2.7 (没有请自行网上查找)
    * 3、安装 pip
        sudo easy_install pip
    * 4、安装 virtualenv
        pip install virtualenv
    * 5、创建项目的虚拟环境
        sh scripts/env_prepare.sh
    * 6、安装依赖包
        备注: 如果是GPU环境,需要将requiremens.txt中的tensorflow替换成tensorflow-gpu
        使用pip install xxx方式安装requiremens.txt下的依赖包


## 项目说明
* classification (分类任务)
    * cnn (卷积神经网络)
        * text_cnn (用于文本分类的TextCNN模型)
            详情见text_cnn下的README.md
    * rnn (循环神经网络)
        * lstm (用于文本分类的lstm/bi-lstm模型)
            详情见lstm下的README.md