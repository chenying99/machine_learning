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
import time
import numpy as np
import tensorflow as tf
from tensorflow.contrib import learn
from constant import PROJECT_DIRECTORY
from data.data_helpers import load_mr_polarity_data_and_labels, batch_iter
from classification.cnn.text_cnn.text_cnn import TextCNN


def train():
    """
    训练函数
    """
    # 1、设置参数
    # num_classes, 分类的类别
    tf.flags.DEFINE_integer('num_classes', 2, 'class num')
    # embedding_dim, 每个词表表示成词向量的长度, 默认为128
    tf.flags.DEFINE_integer("embedding_dim", 128, "Dimensionality of character embedding (default: 128)")
    # filter_sizes, 每个卷积核的高度(宽度为embedding_dim大小), 默认为3,4,5
    tf.flags.DEFINE_string("filter_sizes", "3,4,5", "Comma-separated filter sizes (default: '3,4,5')")
    # num_filters, 每个尺寸下的卷积核的个数, 默认为128
    tf.flags.DEFINE_integer("num_filters", 128, "Number of filters per filter size (default: 128)")
    # dropout_keep_prob, 保留一个神经元的概率，这个概率只在训练的时候用到, 默认为0.5
    tf.flags.DEFINE_float("dropout_keep_prob", 0.5, "Dropout keep probability (default: 0.5)")
    # l2_reg_lambda, L2 正则项的参数lambda,默认为0.0
    tf.flags.DEFINE_float("l2_reg_lambda", 0.0, "L2 regularizaion lambda (default: 0.0)")
    # batch_size, 每批读入样本的数量,默认为64
    tf.flags.DEFINE_integer("batch_size", 64, "Batch Size (default: 64)")
    # max_sentence_length, 文本最大长度,默认为50
    tf.flags.DEFINE_integer("max_sentence_length", 50, "max sentence length")
    # initial_learning_rate, 初始的学习率,默认为0.001
    tf.flags.DEFINE_float('initial_learning_rate', 0.001, 'init learning rate')
    # min_learning_rate, 学习率最小值,默认为0.00001
    tf.flags.DEFINE_float('min_learning_rate', 0.00001, 'min learning rate')
    # decay_rate, 学习率衰减率,默认为0.8
    tf.flags.DEFINE_float('decay_rate', 0.8, 'the learning rate decay')
    # decay_step, 学习率衰减步长,默认为1000
    tf.flags.DEFINE_integer('decay_step', 1000, 'Steps after which learning rate decays')
    # init_scale, 参数随机初始化的最大值,默认为0.1
    tf.flags.DEFINE_float('init_scale', 0.1, 'init scale')
    # num_epochs, 每次训练读取的数据随机的次数,默认为10
    tf.flags.DEFINE_integer("num_epochs", 20, "Number of training epochs (default: 10)")
    # valid_num, 训练数据中, 用于验证数据的数量
    tf.flags.DEFINE_integer('valid_num', 1000, 'num of validation')
    # show_every, 在每个固定迭代次数之后,输出结果
    tf.flags.DEFINE_integer("show_every", 10, "Show train results after this many steps (default: 100)")
    # valid_every, 在每个固定迭代次数之后,在验证数据上评估模型, 默认为100
    tf.flags.DEFINE_integer("valid_every", 100, "Evaluate model on dev set after this many steps (default: 100)")
    # checkpoint_every, 在每个固定迭代次数之后,保存模型, 默认为100
    tf.flags.DEFINE_integer("checkpoint_every", 1000, "Save model after this many steps (default: 100)")
    # out_dir, 在每个固定迭代次数之后,保存模型
    tf.flags.DEFINE_string('out_dir', os.path.join(PROJECT_DIRECTORY, "classification/cnn/text_cnn/data/model"), "The path of saved model")
    # allow_soft_placement, 设置为True时, 如果你指定的设备不存在，允许TF自动分配设备
    tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
    # log_device_placement, 设备上放置操作日志的位置
    tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")

    FLAGS = tf.flags.FLAGS
    FLAGS._parse_flags()
    print("\nParameters:")
    for attr, value in sorted(FLAGS.__flags.items()):
        print("{}={}".format(attr.upper(), value))
    print("")

    class Config(object):
        filter_sizes = list(map(int, FLAGS.filter_sizes.split(",")))
        num_filters = FLAGS.num_filters
        embedding_dim = FLAGS.embedding_dim
        num_classes = FLAGS.num_classes
        dropout_keep_prob = FLAGS.dropout_keep_prob
        initial_learning_rate = FLAGS.initial_learning_rate
        min_learning_rate = FLAGS.min_learning_rate
        decay_rate = FLAGS.decay_rate
        decay_step = FLAGS.decay_step
        batch_size = FLAGS.batch_size
        sentence_length = FLAGS.max_sentence_length
        l2_reg_lambda = FLAGS.l2_reg_lambda

    # 2、数据准备
    # 2.1 加载数据
    print("Loading data...")
    x, y = load_mr_polarity_data_and_labels()

    # 2.2建立词典
    # 获取最大的样本分词长度
    max_sentence_length = max([len(item.split(" ")) for item in x])
    vocab_processor = learn.preprocessing.VocabularyProcessor(max_sentence_length)
    # 将训练数据根据词汇处理器转换成相应的格式
    x = np.array(list(vocab_processor.fit_transform(x)))

    # 2.3随机数据valid_config
    # 设置随机数种子
    np.random.seed(10)
    # 返回一个洗牌后程度为len(y)的数组
    shuffle_indices = np.random.permutation(np.arange(len(y)))
    # 随机后的数据x
    x_shuffled = x[shuffle_indices]
    # 随机后的类别y
    y_shuffled = y[shuffle_indices]
    # 分割训练集和测试集, 用于交叉验证
    valid_sample_index = -FLAGS.valid_num
    x_train, x_valid = x_shuffled[:valid_sample_index], x_shuffled[valid_sample_index:]
    y_train, y_valid = y_shuffled[:valid_sample_index], y_shuffled[valid_sample_index:]
    print("Vocabulary Size: {:d}".format(len(vocab_processor.vocabulary_)))
    print("Train/Valid split: {:d}/{:d}".format(len(y_train), len(y_valid)))

    # 用于训练时的参数
    config = Config()
    config.sentence_length = max_sentence_length
    config.vocabulary_size = len(vocab_processor.vocabulary_)
    # 用于验证时的参数
    valid_config = Config()
    valid_config.vocabulary_size = len(vocab_processor.vocabulary_)
    valid_config.sentence_length = max_sentence_length
    valid_config.dropout_keep_prob = 1.0

    # 3、训练
    print("begin training")
    with tf.Graph().as_default():
        # session配置
        session_conf = tf.ConfigProto(
          allow_soft_placement=FLAGS.allow_soft_placement,
          log_device_placement=FLAGS.log_device_placement)
        # 自定义session然后通过session.as_default() 设置为默认视图
        sess = tf.Session(config=session_conf)
        with sess.as_default():
            initializer = tf.random_uniform_initializer(-1 * FLAGS.init_scale, 1 * FLAGS.init_scale)
            with tf.variable_scope("model", initializer=initializer):
                cnn_model = TextCNN(config=config)

            timestamp = str(int(time.time()))
            out_dir = os.path.abspath(os.path.join(FLAGS.out_dir, "runs", timestamp))
            print("Writing to {}\n".format(out_dir))
            train_summary_writer = tf.summary.FileWriter(os.path.join(out_dir, "summaries", "train"), sess.graph)
            valid_summary_writer = tf.summary.FileWriter(os.path.join(out_dir, "summaries", "valid"), sess.graph)

            checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
            checkpoint_prefix = os.path.join(checkpoint_dir, "model")
            if not os.path.exists(checkpoint_dir):
                os.makedirs(checkpoint_dir)
            # 保存全局变量
            saver = tf.train.Saver(tf.global_variables())
            # 保存字典
            vocab_processor.save(os.path.join(out_dir, "vocab"))
            # 全局变量初始化
            sess.run(tf.global_variables_initializer())

            def train_step(session, model, input_x, input_y, summary_writer=None):
                """
                单一的训练步骤, 定义一个函数用于模型评价、更新批量数据和更新模型参数
                """
                feed_dict = dict()
                feed_dict[model.input_x] = input_x
                feed_dict[model.input_y] = input_y
                feed_dict[model.dropout_keep_prob] = config.dropout_keep_prob
                fetches = [model.train_op, model.global_step, model.learning_rate, model.loss, model.accuracy,
                           model.summary]
                _, global_step, learning_rate_val, loss_val, accuracy_val, summary = session.run(fetches, feed_dict)
                if summary_writer:
                    summary_writer.add_summary(summary, global_step)
                return global_step, loss_val, accuracy_val

            def eval_step(session, model, input_x, input_y, summary_writer=None):
                """
                在验证集上验证模型
                """
                feed_dict = dict()
                feed_dict[model.input_x] = input_x
                feed_dict[model.input_y] = input_y
                feed_dict[model.dropout_keep_prob] = valid_config.dropout_keep_prob
                fetches = [model.global_step, model.loss, model.accuracy, model.summary]
                global_step, loss_val, accuracy_val, summary = session.run(fetches, feed_dict)
                if summary_writer:
                    summary_writer.add_summary(summary, global_step)
                return loss_val, accuracy_val

            for num_epoch in range(FLAGS.num_epochs):
                training_batches = batch_iter(
                    list(zip(x_train, y_train)), FLAGS.batch_size)
                start_time = time.time()
                print ('epoch {}'.format(num_epoch + 1))
                for training_batch in training_batches:
                    x_batch, y_batch = zip(*training_batch)
                    step, _, _ = train_step(sess, cnn_model, x_batch, y_batch)
                    if step % FLAGS.show_every == 0:
                        step_time = (time.time() - start_time) / FLAGS.show_every
                        examples_per_sec = FLAGS.batch_size / step_time
                        _, train_loss, train_accuracy = train_step(sess, cnn_model, x_batch, y_batch, train_summary_writer)
                        learning_rate = cnn_model.learning_rate.eval()
                        print ("Train, epoch {}, step {}, lr {:g}, loss {:g}, acc {:g}, step-time {:g}, examples/sec {:g}"
                        .format(num_epoch + 1, step, learning_rate, train_loss, train_accuracy, step_time,
                                examples_per_sec))
                        start_time = time.time()
                    if step % FLAGS.valid_every == 0:
                        learning_rate = cnn_model.learning_rate.eval()
                        valid_loss, valid_accuracy = eval_step(sess, cnn_model, x_valid, y_valid, valid_summary_writer)
                        print("Valid, step {}, lr {:g}, loss {:g}, acc {:g}".format
                              (step, learning_rate, valid_loss, valid_accuracy))
                    if step % FLAGS.checkpoint_every == 0:
                        path = saver.save(sess, checkpoint_prefix, step)
                        print("Saved model checkpoint to {}\n".format(path))


if __name__ == '__main__':
    train()
