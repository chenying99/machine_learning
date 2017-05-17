#! /usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on 2017年05月15日

@author: MJ
"""
from __future__ import absolute_import
import os
import sys
p = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
if p not in sys.path:
    sys.path.append(p)
import tensorflow as tf
from tensorflow.contrib import learn
import numpy as np
from constant import PROJECT_DIRECTORY, rt_polaritydata_label_list


# checkpoint_dir, 训练时保存的模型
tf.flags.DEFINE_string("checkpoint_dir", os.path.join(PROJECT_DIRECTORY, "classification/cnn/text_cnn/data/model/runs/1495003179/checkpoints"), "Checkpoint directory from training run")
# allow_soft_placement, 设置为True时, 如果你指定的设备不存在，允许TF自动分配设备
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
# log_device_placement, 设备上放置操作日志的位置
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")

# 设置参数
FLAGS = tf.flags.FLAGS
FLAGS._parse_flags()
print("\nParameters:")
for attr, value in sorted(FLAGS.__flags.items()):
    print("{}={}".format(attr.upper(), value))
print("")


def predict(text):
    """
    给定一个文本,预测文本的分类
    """
    index2label_dict = {i: l.strip() for i, l in enumerate(rt_polaritydata_label_list)}
    # 加载词典
    vocab_path = os.path.join(FLAGS.checkpoint_dir, "..", "vocab")
    # 从给定文件恢复词汇处理器。
    vocab_processor = learn.preprocessing.VocabularyProcessor.restore(vocab_path)
    # 将评估数据根据词汇处理器转换成相应的格式
    x_test = np.array(list(vocab_processor.transform([text])))
    # 查找最新保存的检查点文件的文件名
    checkpoint_file = tf.train.latest_checkpoint(FLAGS.checkpoint_dir)
    graph = tf.Graph()
    with graph.as_default():
        # session配置
        session_conf = tf.ConfigProto(
            allow_soft_placement=FLAGS.allow_soft_placement,
            log_device_placement=FLAGS.log_device_placement)
        # 自定义session然后通过session.as_default() 设置为默认视图
        sess = tf.Session(config=session_conf)
        with sess.as_default():
            # 载入保存Meta图
            saver = tf.train.import_meta_graph("{}.meta".format(checkpoint_file))
            # 恢复变量
            saver.restore(sess, checkpoint_file)
            # 从图中根据名称获取占位符
            input_x = graph.get_operation_by_name("model/input_x").outputs[0]
            dropout_keep_prob = graph.get_operation_by_name("model/dropout_keep_prob").outputs[0]
            # 待评估的Tensors
            predictions = graph.get_operation_by_name("model/output/predictions").outputs[0]
            predict_class = sess.run(predictions, {input_x: x_test, dropout_keep_prob: 1.0})[0]
            return index2label_dict.get(predict_class)


if __name__ == '__main__':
    print (predict("a masterpiece four years in the making"))
    print (predict("everything is off."))
