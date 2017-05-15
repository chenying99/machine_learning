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
import numpy as np
import tensorflow as tf
from tensorflow.contrib import learn
import csv
from constant import PROJECT_DIRECTORY
from classification.cnn.text_cnn.data_helpers import load_data_and_labels, batch_iter


def eval():
    """
    评估函数
    """
    # positive_data_file, 正样本训练数据所在路径
    tf.flags.DEFINE_string("positive_data_file", os.path.join(PROJECT_DIRECTORY, "classification/cnn/text_cnn/data/rt-polaritydata/rt-polarity.pos"), "Data source for the positive data.")
    # negative_data_file, 负样本训练数据所在路径
    tf.flags.DEFINE_string("negative_data_file", os.path.join(PROJECT_DIRECTORY, "classification/cnn/text_cnn/data/rt-polaritydata/rt-polarity.neg"), "Data source for the positive data.")

    # 评估参数
    # batch_size, 每批读入样本的数量,默认为64
    tf.flags.DEFINE_integer("batch_size", 64, "Batch Size (default: 64)")
    # checkpoint_dir文件夹, 训练时保存的
    tf.flags.DEFINE_string("checkpoint_dir", os.path.join(PROJECT_DIRECTORY, "classification/cnn/text_cnn/data/model/runs/1494832207/checkpoints"), "Checkpoint directory from training run")
    # eval_train,是否使用训练数据进行评估
    tf.flags.DEFINE_boolean("eval_train", True, "Evaluate on all training data")

    # Misc Parameters
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

    # 加载数据, 在此可以加载自己的数据
    if FLAGS.eval_train:
        x_raw, y_test = load_data_and_labels(FLAGS.positive_data_file, FLAGS.negative_data_file)
        y_test = np.argmax(y_test, axis=1)
    else:
        x_raw = ["a masterpiece four years in the making", "everything is off."]
        y_test = [1, 0]

    vocab_path = os.path.join(FLAGS.checkpoint_dir, "..", "vocab")
    # 从给定文件恢复词汇处理器。
    vocab_processor = learn.preprocessing.VocabularyProcessor.restore(vocab_path)
    # 将评估数据根据词汇处理器转换成相应的格式
    x_test = np.array(list(vocab_processor.transform(x_raw)))

    print("\nEvaluating...\n")

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
            batches = batch_iter(list(x_test), FLAGS.batch_size, shuffle=False)
            all_predictions = []
            for x_test_batch in batches:
                batch_predictions = sess.run(predictions, {input_x: x_test_batch, dropout_keep_prob: 1.0})
                all_predictions = np.concatenate([all_predictions, batch_predictions])

    if y_test is not None:
        # 预测正确的样本个数
        correct_predictions = float(sum(all_predictions == y_test))
        # 输出预测正确样本的个数
        print("Total number of test examples: {}".format(len(y_test)))
        # 输出预测正确率
        print("Accuracy: {:g}".format(correct_predictions/float(len(y_test))))

    # 将数据和预测结果合并成两列
    predictions_human_readable = np.column_stack((np.array(x_raw), all_predictions))
    out_path = os.path.join(FLAGS.checkpoint_dir, "..", "prediction.csv")
    # 保存结果到csv文件
    print("Saving evaluation to {0}".format(out_path))
    with open(out_path, 'w') as f:
        csv.writer(f).writerows(predictions_human_readable)


if __name__ == '__main__':
    eval()
