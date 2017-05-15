# Created on 2017年05月15日

# author: MJ

# project: machine_learning

# 中文文档

## 项目依赖
1、IDE PyCharm Community Edition 2016.1.5
2、安装 python 2.7 (没有请自行网上查找)
3、安装 pip
   sudo easy_install pip
4、安装 virtualenv
   pip install virtualenv
5、安装环境
   sh scripts/env_prepare.sh
6、更新环境
   备注: 如果是GPU环境,需要将requiremens.txt中的tensorflow替换成tensorflow-gpu
   sh scripts/env_update.sh


## 项目说明
* classification (分类任务)
    * cnn (卷积神经网络)
        * text_cnn (用于文本分类的TextCNN模型)
            详情见text_cnn下的README.md

