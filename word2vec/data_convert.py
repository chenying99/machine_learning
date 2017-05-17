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
import numpy as np
from word2vec.word2vec_by_gensim_utils import get_word_2_vec_by_gensim_for_sogou_classification
from constant import MAX_DOCUMENT_LENGTH


def default_tokenizer():
    return lambda x: x.split()


class SimpleTextConverter(object):

    def __init__(self, word_vec, max_document_length, tokenizer_fn=None):
        self.syn0norm = word_vec.syn0norm
        self.vocab = word_vec.vocab
        self.tokenizer_fn = tokenizer_fn or default_tokenizer()

        self.max_document_length = max_document_length

    def transform_to_ids(self, raw_documents):
            for text in raw_documents:
                tokens = self.tokenizer_fn(text)
                word_ids = np.zeros(self.max_document_length, np.int64)

                idx = 0
                for token in tokens:
                    if token not in self.vocab:
                        continue
                    if idx >= self.max_document_length:
                        break
                    word_ids[idx] = self.vocab[token].index
                    idx += 1
                yield word_ids, idx


text_converter_for_sogou_classification = None


def get_text_converter_for_sogou_classification(max_document_length=MAX_DOCUMENT_LENGTH, tokenizer_fn=None):
    """
    由于使用全局变量, 第一次初始化后tokenizer_fn不可更新
    """
    global text_converter_for_sogou_classification
    if not text_converter_for_sogou_classification:
        word_vec = get_word_2_vec_by_gensim_for_sogou_classification()
        text_converter_for_sogou_classification = SimpleTextConverter(word_vec, max_document_length, tokenizer_fn)
    return text_converter_for_sogou_classification
