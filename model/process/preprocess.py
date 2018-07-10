#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import codecs
import re
import spacy
import fasttext
from collections import Counter

nlp = spacy.load('en')


def textpre_clean(text):
    text = re.sub("[¡\!\?\¿.,:\"Ü\-\(\)\[\]\{\}\_\/\\@#$%&——]", "", text)
    words = text.split()
    stop_words = nlp.Defaults.stop_words
    clean_words = []
    for word in words:
        word = word.lower()
        if re.match("[\d]", word):
            continue
        elif word in stop_words:
            continue
        else:
            clean_words.append(word)
    test_doc = nlp(text)
    token_stemmed = []
    for token in test_doc:
        token_stemmed.append(token.lemma_)
    stemmed_sentence = " ".join(token_stemmed)
    return stemmed_sentence


def load_train_data():
    file_train = "../../data/cikm_spanish_train_20180516.txt"
    f = codecs.open(file_train, "r", "utf-8")
    sen = []
    label = []
    ft_train = "../../data/v1.0/ft_train.txt"
    ft_train = codecs.open(ft_train, "w", "utf-8")
    for line in f:
        line_list = line.split("/t")
        sen1 = line_list[0]
        sen2 = line_list[2]
        sen = sen1 + ' ' + sen2
        sen = textpre_clean(sen)
        label.append(line_list[4])
        label = "__label__" + str(line_list[4])
        ft_train.write(sen, label)
    ft_train.close()
    f.close()
    counter = Counter(label)
    print("counter:",counter)

def load_test_data():
    file_test = "../../data/cikm_test_a_20180516.txt"
    f = codecs.open(file_test, "r", "utf-8")
    ft_test = "../../data/v1.0/ft_test.txt"
    ft_test = codecs.open(ft_test, "w", "utf-8")
    for line in f:
        line_list = line.split("/t")
        sen1 = line_list[0]
        sen2 = line_list[1]
        sen = sen1 + ' ' + sen2
        sen = textpre_clean(sen)
        ft_test.write(sen)
    ft_test.close()
    f.close()

def train_model():
    classifier=fasttext.supervised(input = "../../data/v1.0/ft_train.txt", wordNgrams = 2, ws =8, epoch = 10)
    return classifier

def predict():
    classifier = train_model()
    ft_test = "../../data/v1.0/ft_test.txt"
    ft_test = codecs.open(ft_test, "r", "utf-8")
    file_out = "../../data/v1.0/re.txt"
    f_re = open(file_out,"w")
    for each in ft_test:
        re = classifier.predict(each, k=2)
        labels = re[0]
        probas = re[2]
        for i in range(0, 2):
            if labels[i] == '__label__1':
                score = probas[i]
                r = str(score)+"\n"
                f_re.write(r)
    ft_test.close()
    f_re.close()



# file_vec = "../../data/fast_text_vectors_wiki.es.vec/wiki.es.vec"
# load_vec(file_vec)

load_train_data()

"""
#转换为小写字符串
def textpre_lower(text):
    return text.lower()

#删除数字，或替换为number
def textpre_num(text, replace=False):
    words = text.split()
    nonum_words = []
    for word in words:
        if re.match("[\d]", word):
            if replace == False:
                continue
            else:
                nonum_words.append("número")
        else:
            nonum_words.append(word)
    return " ".join(nonum_words)

#词干化
def textpre_stem(text):
    test_doc = nlp(text)
    token_stemmed = []
    for token in test_doc:
        token_stemmed.append(token.lemma_)
    stemmed_sentence = " ".join(token_stemmed)
    return stemmed_sentence

#POS表示
def textpre_pos(text):
    test_doc = nlp(text)
    token_pos = []
    for token in test_doc:
        token_pos.append(token.pos_)
    pos_sentence = " ".join(token_pos)
    return pos_sentence

#去除标点符号
def textpre_punction(text):
    text_clean = re.sub("[¡\!\?\¿.,:\"Ü\-\(\)\[\]\{\}\_\/\\@#$%&——]", "", text)
    return text_clean

#去除停用词
def textpre_stopwords(text):
    stop_words = nlp.Defaults.stop_words
    words = text.split()
    nostop_words = []
    for word in words:
        if word in stop_words:
            continue
        else:
            nostop_words.append(word)
    return " ".join(nostop_words)
"""
