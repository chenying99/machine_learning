#!/usr/bin/python
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
import numpy as np
import re
from constant import PROJECT_DIRECTORY, rt_polaritydata_label_list, sogou_classification_label_list
from data.prepare import segment_sogou_classification_data
from word2vec.data_convert import get_text_converter_for_sogou_classification


def clean_str(string):
    """
    Tokenization/string cleaning for all datasets except for SST.
    Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
    """
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower()


def load_mr_polarity_data_and_labels():
    """
    加载电影评论数据(Movie Review data from Rotten Tomatoes),包含5331个积极评论和5331个消极评论
    """
    # positive_data_file, 正样本训练数据所在路径
    positive_data_file = os.path.join(PROJECT_DIRECTORY, "data/rt-polaritydata/rt-polarity.pos")
    # negative_data_file, 负样本训练数据所在路径
    negative_data_file = os.path.join(PROJECT_DIRECTORY, "data/rt-polaritydata/rt-polarity.neg")
    positive_examples = list(open(positive_data_file, "r").readlines())
    positive_examples = [s.strip() for s in positive_examples]
    negative_examples = list(open(negative_data_file, "r").readlines())
    negative_examples = [s.strip() for s in negative_examples]
    # 分词
    x_text = positive_examples + negative_examples
    x_text = [clean_str(sent) for sent in x_text]
    # 生成label
    label2index_dict = {l.strip(): i for i, l in enumerate(rt_polaritydata_label_list)}
    label_item = np.zeros(len(rt_polaritydata_label_list), np.float32)
    label_item[label2index_dict['pos']] = 1
    positive_labels = [label_item for _ in positive_examples]
    label_item = np.zeros(len(rt_polaritydata_label_list), np.float32)
    label_item[label2index_dict['neg']] = 1
    negative_labels = [label_item for _ in negative_examples]
    y = np.concatenate([positive_labels, negative_labels], 0)
    return [x_text, y]


def load_sogou_classification_data_and_labels(max_sentence_length=None):
    """
    从本地文件读取搜狗分类数据集
    """
    read_dir_path = os.path.join(PROJECT_DIRECTORY, "data/sogou_classification_segment_data")
    if not os.path.exists(read_dir_path):
        segment_sogou_classification_data()
    label_dir_list = os.listdir(read_dir_path)
    x_raw = []
    y = []
    label2index_dict = {l.strip(): i for i, l in enumerate(sogou_classification_label_list)}
    for label_dir in label_dir_list:
        label_dir_path = os.path.join(read_dir_path, label_dir)
        label_index = label2index_dict[label_dir]
        label_item = np.zeros(len(sogou_classification_label_list), np.float32)
        label_item[label_index] = 1
        label_file_list = os.listdir(label_dir_path)
        for label_file in label_file_list:
            with open(os.path.join(label_dir_path, label_file), 'r') as reader:
                text = reader.read().replace('\n', '').replace('\r', '').strip()
                x_raw.append(text)
                y.append(label_item)
    if not max_sentence_length:
        max_sentence_length = max([len(item.split(" ")) for item in x_raw])
    x = []
    converter = get_text_converter_for_sogou_classification(max_sentence_length)
    for doc, doc_len in converter.transform_to_ids(x_raw):
        x.append(doc)

    return np.array(x), np.array(y), max_sentence_length


def batch_iter(data, batch_size, shuffle=True):
    """
    Generates a batch iterator for a dataset.
    为数据集生成批迭代器
    """
    data = np.array(data)
    data_size = len(data)
    num_batches_per_epoch = int(len(data)/batch_size) + 1
    # Shuffle the data at each epoch
    if shuffle:
        shuffle_indices = np.random.permutation(np.arange(data_size))
        shuffled_data = data[shuffle_indices]
    else:
        shuffled_data = data
    for batch_num in range(num_batches_per_epoch):
        start_index = batch_num * batch_size
        end_index = min((batch_num + 1) * batch_size, data_size)
        yield shuffled_data[start_index:end_index]
