#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
Created on 2017年05月15日

@author: MJ
"""
import tensorflow as tf


class TextCNN(object):
    """
    搭建一个用于文本数据的CNN模型,使用嵌入层(embedding layer)，其次是卷积层(convolutional)，最大池层(max-pooling)和softmax层。
    """
    def __init__(self, config):
        """
        初始化
        """
        self.filter_sizes = config.filter_sizes
        self.initial_learning_rate = config.initial_learning_rate
        self.min_learning_rate = config.min_learning_rate
        self.decay_rate = config.decay_rate
        self.decay_step = config.decay_step
        self.num_step = config.num_step
        self.num_classes = config.num_classes
        self.num_filters = config.num_filters
        self.vocabulary_size = config.vocabulary_size
        self.embedding_dim = config.embedding_dim
        self.l2_reg_lambda = config.l2_reg_lambda
        # 输入占位符，输出占位符和dropout占位符
        self.input_x = tf.placeholder(tf.int32, [None, self.num_step], name="input_x")
        self.input_y = tf.placeholder(tf.int32, [None, self.num_classes], name="input_y")
        # dropout_keep_prob是保留一个神经元的概率，这个概率只在训练的时候用到
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")
        # L2正规化损失记录（可选）
        l2_loss = tf.constant(0.0)
        # 嵌入层
        # tf.device('/cpu:0')表示使用cpu进行操作，因为tensorflow当gpu可用时默认使用gpu，但是embedding不支持gpu实现，所以使用CPU操作
        # tf.name_scope("embedding"),把所有操作加到命名为embedding的顶层节点，用于可视化网络视图
        with tf.device('/cpu:0'), tf.name_scope("embedding"):
            # W是在训练时得到的嵌入矩阵，通过随机均匀分布进行初始化
            W = tf.Variable(tf.random_uniform([self.vocabulary_size,  self.embedding_dim]), name="W")
            # tf.nn.embedding_lookup是真正的embedding操作, 查找input_x中所有的ids，获取它们的word vector。batch中的每个sentence的每个word都要查找。
            # 所以得到的结果是一个三维的tensor，[None, sequence_length, embedding_size]
            self.embedded_chars = tf.nn.embedding_lookup(W, self.input_x)
            # 因为卷积操作conv2d的input要求4个维度的tensor, 所以需要给embedding结果增加一个维度来适应conv2d的input要求
            # 传入的-1表示在最后位置插入, 得到[None, sequence_length, embedding_size, 1]
            self.embedded_chars_expanded = tf.expand_dims(self.embedded_chars, -1)

        # 对不同大小的filter建立不同的卷积层+最大池层
        # pooled_outputs用于存储池化之后的结果,用于后面的全连接层
        pooled_outputs = []
        for i, filter_size in enumerate(self.filter_sizes):
            with tf.name_scope("conv-maxpool-%s" % filter_size):
                # 卷积层
                filter_shape = [filter_size, self.embedding_dim, 1, self.num_filters]
                # W是卷积的输入矩阵
                # 利用truncated_normal生成截断正态分布随机数, 尺寸是filter_shape, 均值mean, 标准差stddev,
                # 不过只保留[mean-2*stddev,mean+2*stddev]范围内的随机数
                W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W")
                # b是卷积的输入偏置量
                b = tf.Variable(tf.constant(0.1, shape=[self.num_filters]), name="b")
                # 卷积操作, “VALID”表示使用narrow卷积，得到的结果大小为[batch, sequence_length - filter_size + 1, 1, num_filters]
                conv = tf.nn.conv2d(
                    self.embedded_chars_expanded,
                    W,
                    strides=[1, 1, 1, 1],
                    padding="VALID",
                    name="conv")
                # h: 卷积的结果加上偏置项b，之后应用ReLU函数处理的结果
                # h的大小为[batch, sequence_length - filter_size + 1, 1, num_filters]
                h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")
                # 用max-pooling处理上层的输出结果,每一个卷积结果
                # pooled的大小为[batch, 1, 1, num_filters]，
                pooled = tf.nn.max_pool(
                    h,
                    ksize=[1, self.num_step - filter_size + 1, 1, 1],
                    strides=[1, 1, 1, 1],
                    padding='VALID',
                    name="pool")
                pooled_outputs.append(pooled)

        # 全连接输出层
        # 将上面的pooling层输出全连接到输出层
        num_filters_total = self.num_filters * len(self.filter_sizes)
        # 把相同filter_size的所有pooled结果concat起来，再将不同的filter_size之间的结果concat起来
        # tf.concat按某一维度进行合并, h_pool的大小为[batch, 1, 1, num_filters_total]
        self.h_pool = tf.concat(pooled_outputs, 3)
        # h_pool_flat也就是[batch, num_filters_total]维的tensor。
        self.h_pool_flat = tf.reshape(self.h_pool, [-1, num_filters_total])

        # 在训练阶段，对max-pooling layer的输出实行一些dropout，以概率p激活，激活的部分传递给softmax层。
        with tf.name_scope("dropout"):
            self.h_drop = tf.nn.dropout(self.h_pool_flat, self.dropout_keep_prob)

        # 输出层
        with tf.name_scope("output"):
            W = tf.get_variable(
                "W",
                shape=[num_filters_total, self.num_classes],
                initializer=tf.contrib.layers.xavier_initializer())
            b = tf.Variable(tf.constant(0.1, shape=[self.num_classes]), name="b")
            l2_loss += tf.nn.l2_loss(W)
            l2_loss += tf.nn.l2_loss(b)
            # 得到所有类别的分数, scores的shape为[batch, num_classes]
            self.scores = tf.nn.xw_plus_b(self.h_drop, W, b, name="scores")
            # 计算预测类别,分数最大对应的类别，因此argmax的时候是选取每行的max，dimention=1,[batch, 1]
            self.predictions = tf.argmax(self.scores, 1, name="predictions")

        # 计算交叉熵的平均损失
        with tf.name_scope("loss"):
            losses = tf.nn.softmax_cross_entropy_with_logits(labels=self.input_y, logits=self.scores)
            # 为了防止过拟合，最后还要在loss func中加入l2正则项，即l2_loss。l2_reg_lambda来确定惩罚的力度
            self.loss = tf.reduce_mean(losses) + self.l2_reg_lambda * l2_loss
            tf.summary.scalar("loss", self.loss)

        # 计算精度
        with tf.name_scope("accuracy"):
            # tf.equal(x, y)返回的是一个bool tensor，如果xy对应位置的值相等就是true，否则false。得到的tensor是[batch, 1]的。
            correct_predictions = tf.equal(self.predictions, tf.argmax(self.input_y, 1))
            # tf.cast(x, dtype)将bool tensor转化成float类型的tensor，方便计算
            # tf.reduce_mean()本身输入的就是一个float类型的vector（元素要么是0.0，要么是1.0），
            # 直接对这样的vector计算mean得到的就是accuracy，不需要指定reduction_indices
            self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")
            tf.summary.scalar("accuracy", self.accuracy)

        self.global_step = tf.Variable(0, name="global_step", trainable=False)

        self.learning_rate = tf.maximum(tf.train.exponential_decay(self.initial_learning_rate, self.global_step,
                                                                   self.decay_step,
                                                                   self.decay_rate, staircase=True),
                                        self.min_learning_rate)
        tf.summary.scalar("learning_rate", self.learning_rate)

        optimizer = tf.train.AdamOptimizer(self.learning_rate)
        grads_and_vars = optimizer.compute_gradients(self.loss)
        self.train_op = optimizer.apply_gradients(grads_and_vars, global_step=self.global_step)

        grad_summaries = []
        for grad, var in grads_and_vars:
            if grad is not None:
                grad_hist_summary = tf.summary.histogram("{}/grad/hist".format(var.name), grad)
                sparsity_summary = tf.summary.scalar("{}/grad/sparsity".format(var.name), tf.nn.zero_fraction(grad))
                grad_summaries.append(grad_hist_summary)
                grad_summaries.append(sparsity_summary)
        self.summary = tf.summary.merge_all()

