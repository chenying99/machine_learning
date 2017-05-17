#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
Created on 2017年05月16日

@author: MJ
"""
from __future__ import absolute_import
import os
import sys
p = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if p not in sys.path:
    sys.path.append(p)
from constant import PROJECT_DIRECTORY
from word2vec.word2vec_by_gensim_utils import train_word2vec_by_gensim


def train_sogou_classification_word2vec():
    """
    训练用于搜狗分类语料的word2vec
    """
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


if __name__ == '__main__':
    train_sogou_classification_word2vec()
