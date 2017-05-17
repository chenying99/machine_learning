#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
Created on 2017年5月16日

@author: MJ
"""
from __future__ import absolute_import
import os
import sys
p = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if p not in sys.path:
    sys.path.append(p)
import time
import numpy as np
from gensim.models import Word2Vec
from gensim import matutils
from constant import PROJECT_DIRECTORY, WORD_2_VEC_MODEL


word_2_vec_by_gensim = None
word_2_vec_by_gensim_for_sogou_classification = None


def train_word2vec_by_gensim(sentences, model_name):
    """
    训练word2vec
    :param sentences: 分词列表,例如[['first', 'sentence'], ['second', 'sentence']]
    :param model_name: 保存Word2Vec的模型名
    :return:
    """
    print ('Start train_word2vec_by_gensim ...')
    start = time.clock()
    print ('model name: %s' % model_name)
    model = Word2Vec(sentences=sentences, max_vocab_size=None, window=8, size=256, min_count=5, workers=4, iter=20)
    # 保存模型，以便重用
    model.save(get_word_2_vec_path_by_gensim(model_name))
    end = time.clock()
    print ('Completed, time consuming: %f s' % (end - start))


def get_word_2_vec_path_by_gensim(model_name):
    """
    获取word2vec模型所在路径
    """
    model_directory = os.path.join(PROJECT_DIRECTORY, WORD_2_VEC_MODEL, model_name)
    if not os.path.exists(model_directory):
        os.makedirs(model_directory)
    path = os.path.join(model_directory, 'word2vec')
    return path


def get_word_2_vec_by_gensim(model_name):
    """
    获取model_name对应的word2vec
    """
    global word_2_vec_by_gensim
    if not word_2_vec_by_gensim:
        word2vec_path = get_word_2_vec_path_by_gensim(model_name)
        # 从文件加载Word2Vec模型
        word_2_vec_by_gensim = Word2Vec.load(word2vec_path)
        # 预计算L2归一化后的向量,如果replace为True, 丢弃原来的载体，只保存归一化后的.这样可以节省大量的内存, 做替换该模型不能继续训练,变成只读的
        word_2_vec_by_gensim.init_sims(replace=True)
    return word_2_vec_by_gensim


def get_word_2_vec_by_gensim_for_sogou_classification():
    """
    获取model_name对应的word2vec
    """
    global word_2_vec_by_gensim_for_sogou_classification
    if not word_2_vec_by_gensim_for_sogou_classification:
        word2vec_path = get_word_2_vec_path_by_gensim('word2vec_for_sogou_classification')
        # 从文件加载Word2Vec模型
        model = Word2Vec.load(word2vec_path)
        # 预计算L2归一化后的向量,如果replace为True, 丢弃原来的载体，只保存归一化后的.这样可以节省大量的内存, 做替换该模型不能继续训练,变成只读的
        model.init_sims(replace=True)
        # 补充一个特殊向量放到 word2vec 的第一个,并把 word2vec 的第一个词向量挪到最后一个
        # 找出下标中的第一个词
        first_word = model.index2word[0]
        # 把第一个词的下标换成词的个数,这样没有添加新的词向量之前这个下标是不存在的,添加了之后位置就对了
        model.vocab[first_word].index = len(model.vocab)
        # 重新设置 word2vec 的词向量矩阵为特殊向量 + 第一个词向量之后的向量 + 第一个词向量
        model.syn0norm = np.concatenate(
            (matutils.unitvec(np.sum(model.syn0norm, axis=0, keepdims=True) / model.syn0norm.shape[0]),
             model.syn0norm[1:],
             model.syn0norm[:1]))
        word_2_vec_by_gensim_for_sogou_classification = model
    return word_2_vec_by_gensim_for_sogou_classification


def get_word_most_similar_words_in_word_2_vec_by_gensim(model_name, word, top=20):
    """
    计算某个词的相关词列表
    """
    model = get_word_2_vec_by_gensim(model_name)
    return model.most_similar(word, topn=top)


def get_similarity_between_two_words_in_word_2_vec_by_gensim(model_name, word_1, word_2):
    """
    计算两个词的相似度/相关程度
    """
    model = get_word_2_vec_by_gensim(model_name)
    return model.similarity(word_1, word_2)


def get_similarity_between_two_sentence_in_word_2_vec_by_gensim(model_name, sentence_1, sentence_2):
    """
    计算两个句子之间的相似度/相关程度
    """
    model = get_word_2_vec_by_gensim(model_name)
    return model.n_similarity(sentence_1, sentence_2)


if __name__ == '__main__':
    sentences = []
    read_dir_path = os.path.join(PROJECT_DIRECTORY, "data/sogou_classification_segment_data")
    label_dir_list = os.listdir(read_dir_path)
    for label_dir in label_dir_list:
        label_dir_path = os.path.join(read_dir_path, label_dir)
        label_file_list = os.listdir(label_dir_path)
        for label_file in label_file_list:
            with open(os.path.join(label_dir_path, label_file), 'r') as reader:
                word_list = reader.read().replace('\n', '').replace('\r', '').strip().split()
                sentences.append(word_list)
    model_name = 'word2vec_for_sogou_classification'
    train_word2vec_by_gensim(sentences, model_name)
