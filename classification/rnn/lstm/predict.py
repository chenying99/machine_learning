#! /usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on 2017年05月16日

@author: MJ
"""
from __future__ import absolute_import
import os
import sys
p = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
if p not in sys.path:
    sys.path.append(p)
import tensorflow as tf
import numpy as np
import jieba
from constant import PROJECT_DIRECTORY, sogou_classification_label_list
from data.prepare import get_sogou_classification_stopwords_set
from utils.utils import ensure_unicode
from word2vec.data_convert import get_text_converter_for_sogou_classification


# checkpoint_dir, 训练时保存的模型
tf.flags.DEFINE_string("checkpoint_dir", os.path.join(PROJECT_DIRECTORY, "classification/rnn/lstm/data/model/runs/1494832207/checkpoints"), "Checkpoint directory from training run")
# max_sentence_length, 文本最大长度
tf.flags.DEFINE_integer("max_sentence_length", 500, "max sentence length")
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


def predict_doc(text):
    """
    给定一个文本,预测文本的分类
    """
    text = ensure_unicode(text)
    stopwords_set = get_sogou_classification_stopwords_set()
    segment_list = jieba.cut(text)
    word_list = []
    for word in segment_list:
        word = word.strip()
        if '' != word and word not in stopwords_set:
            word_list.append(word)
    word_segment = ' '.join(word_list)

    # 查找最新保存的检查点文件的文件名
    checkpoint_dir = FLAGS.checkpoint_dir
    checkpoint_file = tf.train.latest_checkpoint(checkpoint_dir)
    index2label_dict = {i: l.strip() for i, l in enumerate(sogou_classification_label_list)}
    converter = get_text_converter_for_sogou_classification(FLAGS.max_sentence_length)
    x_test = []
    for doc, _ in converter.transform_to_ids([word_segment]):
        x_test.append(doc)
    x_test = np.array(x_test)
    with tf.Graph().as_default() as graph:
        # session配置
        session_conf = tf.ConfigProto(
            allow_soft_placement=True,
            log_device_placement=False)
        # 自定义session然后通过session.as_default() 设置为默认视图
        with tf.Session(config=session_conf).as_default() as sess:
            # 载入保存Meta图
            saver = tf.train.import_meta_graph("{}.meta".format(checkpoint_file))
            # 恢复变量
            saver.restore(sess, checkpoint_file)
            # 从图中根据名称获取占位符
            input_x = graph.get_operation_by_name("model/input_x").outputs[0]

            dropout_keep_prob = graph.get_operation_by_name("model/dropout_keep_prob").outputs[0]

            # 待评估的Tensors
            prediction = graph.get_operation_by_name("model/output/prediction").outputs[0]
            predict_class = sess.run(prediction, {input_x: x_test, dropout_keep_prob: 1.0})[0]
            return index2label_dict.get(predict_class)


if __name__ == '__main__':
    print (predict_doc("本报讯 (记者 王京) 联想THINKPAD近期几乎全系列笔记本电脑降价促销，最高降幅达到800美元，降幅达到42%。这是记者昨天从联想美国官方网站发现的。联想相关人士表示，这是为纪念新联想成立1周年而在美国市场推出的促销，产品包括THINKPAD T、X以及Z系列笔记本。促销不是打价格战，THINK品牌走高端商务路线方向不会改变。"))
