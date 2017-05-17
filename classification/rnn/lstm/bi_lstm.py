#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
Created on 2017年05月16日

@author: MJ
"""
import tensorflow as tf


def variable_summaries(var):
    """
    添加summaries
    """
    print (var.op.name, ' ', var.get_shape().as_list())


class BiLSTM(object):
    """
    基于BiLSTM的文本分类模型
    """
    def __init__(self, config):
        """
        初始化
        """
        self.initial_learning_rate = config.initial_learning_rate
        self.min_learning_rate = config.min_learning_rate
        self.decay_step = config.decay_step
        self.decay_rate = config.decay_rate
        self.num_step = config.num_step
        self.num_classes = config.num_classes
        self.hidden_neural_size = config.hidden_neural_size
        self.vocabulary_size = config.vocabulary_size
        self.embedding_dim = config.embedding_dim
        self.hidden_layer_num = config.hidden_layer_num
        # 输入占位符，输出占位符和dropout占位符
        self.input_x = tf.placeholder(tf.int32, [None, self.num_step], name="input_x")
        self.input_y = tf.placeholder(tf.int32, [None, self.num_classes], name="input_y")
        # dropout_keep_prob是保留一个神经元的概率，这个概率只在训练的时候用到
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")
        variable_summaries(self.input_x)

        # embedding层
        with tf.device("/cpu:0"), tf.name_scope("embedding_layer"):
            # W是在训练时得到的嵌入矩阵，通过随机均匀分布进行初始化
            W = tf.Variable(tf.random_uniform([self.vocabulary_size, self.embedding_dim]), name="W")
            # tf.nn.embedding_lookup是真正的embedding操作, 查找input_x中所有的ids，获取它们的word vector。batch中的每个sentence的每个word都要查找。
            # 所以得到的结果是一个三维的tensor，[None, sequence_length, embedding_size]
            inputs = tf.nn.embedding_lookup(W, self.input_x)
            # 在训练阶段，对inputs实行一些dropout
            inputs = tf.nn.dropout(inputs, self.dropout_keep_prob, name="dropout")
            # 输出inputs的shape, 方便调试
            variable_summaries(inputs)

            # 对数据进行一些处理,把形状为(batch_size, n_steps, n_input)的输入变成长度为n_steps的列表,
            # 而其中元素形状为(batch_size, n_input), 这样符合LSTM单元的输入格式
            # 1、将第一个维度 batch_size 和第二个维度 n_steps进行交换
            inputs = tf.transpose(inputs, [1, 0, 2])
            # 2、变形为(n_steps*batch_size, n_input)的形状
            inputs = tf.reshape(inputs, [-1, self.embedding_dim])
            # 3、将x拆成长度为n_steps的列表,列表中的每个tensor的尺寸都是(batch_size, n_input)
            inputs = tf.split(inputs, self.num_step, 0)
            variable_summaries(self.input_x)

        def fw_lstm_cell():
            # Forward的LSTM单元
            lstm_cell = tf.contrib.rnn.LSTMCell(self.hidden_neural_size, forget_bias=2.0)
            lstm_cell = tf.contrib.rnn.DropoutWrapper(lstm_cell, output_keep_prob=self.dropout_keep_prob)
            return lstm_cell

        def bw_lstm_cell():
            # Backward的LSTM单元
            lstm_cell = tf.contrib.rnn.LSTMCell(self.hidden_neural_size, forget_bias=2.0)
            lstm_cell = tf.contrib.rnn.DropoutWrapper(lstm_cell, output_keep_prob=self.dropout_keep_prob)
            return lstm_cell

        if self.hidden_layer_num > 1:
            fw_lstm_cells = tf.contrib.rnn.MultiRNNCell([fw_lstm_cell() for _ in range(self.hidden_layer_num)], state_is_tuple=True)
            bw_lstm_cells = tf.contrib.rnn.MultiRNNCell([bw_lstm_cell() for _ in range(self.hidden_layer_num)], state_is_tuple=True)
        else:
            fw_lstm_cells = fw_lstm_cell()
            bw_lstm_cells = bw_lstm_cell()

        # 获取LSTM单元输出outputs
        outputs, _, _ = tf.contrib.rnn.static_bidirectional_rnn(fw_lstm_cells, bw_lstm_cells, inputs, dtype=tf.float32)

        with tf.name_scope("mean_pooling_layer"):
            # LSTM单元最后的输出结果output做一个矩阵乘法并加上偏置
            output = outputs[-1]
            variable_summaries(output)

        with tf.name_scope("softmax_layer"):
            softmax_w = tf.get_variable("softmax_w", [2*self.hidden_neural_size, self.num_classes], dtype=tf.float32)
            softmax_b = tf.get_variable("softmax_b", [self.num_classes], dtype=tf.float32)
            # 得到所有类别的分数
            self.logits = tf.add(tf.matmul(output, softmax_w), softmax_b, name="logits")
            variable_summaries(self.logits)

        # 输出层
        with tf.name_scope("output"):
            self.cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=self.input_y, logits=self.logits + 1e-10)
            self.loss = tf.reduce_mean(self.cross_entropy, name="loss")
            tf.summary.scalar("loss", self.loss)
            # 计算预测类别,分数最大对应的类别
            self.prediction = tf.argmax(self.logits, 1, name="prediction")
            # tf.equal(x, y)返回的是一个bool tensor，如果xy对应位置的值相等就是true，否则false。得到的tensor是[batch, 1]的。
            correct_prediction = tf.equal(self.prediction,  tf.argmax(self.input_y, 1))
            self.correct_num = tf.reduce_sum(tf.cast(correct_prediction, tf.float32))
            # tf.cast(x, dtype)将bool tensor转化成float类型的tensor，方便计算
            # tf.reduce_mean()本身输入的就是一个float类型的vector（元素要么是0.0，要么是1.0），
            # 直接对这样的vector计算mean得到的就是accuracy，不需要指定reduction_indices
            self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32), name="accuracy")
            tf.summary.scalar("accuracy", self.accuracy)

        self.global_step = tf.Variable(0, name="global_step", trainable=False)

        self.learning_rate = tf.maximum(tf.train.exponential_decay(self.initial_learning_rate, self.global_step,
                                                                   self.decay_step,
                                                                   self.decay_rate, staircase=True),
                                        self.min_learning_rate)
        tf.summary.scalar("learning_rate", self.learning_rate)

        tvars = tf.trainable_variables()
        grads, _ = tf.clip_by_global_norm(tf.gradients(self.loss, tvars), config.max_grad_norm)

        optimizer = tf.train.AdamOptimizer(self.learning_rate)
        optimizer.apply_gradients(zip(grads, tvars))
        self.train_op = optimizer.apply_gradients(zip(grads, tvars), global_step=self.global_step)
        self.summary = tf.summary.merge_all()

