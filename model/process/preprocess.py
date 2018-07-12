#!/usr/bin/env python
# -*- coding: utf-8 -*-

import nltk
import re
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
import codecs
from collections import Counter

class preprocess(object):
    def __init__(self, conf):
        self.file_train = conf["file_train"]
        self.file_test = conf["file_test"]
        self.file_ft_train = conf["file_ft_train"]
        self.file_ft_test = conf["file_ft_test"]

    def clean_str(self, text):
        snowball_stemmer = SnowballStemmer("spanish")
        text = re.sub("[¡\!\?\¿.,:\"Ü\-\(\)\[\]\{\}\_\/\\@#$%&——]", "", text)
        text_list = text.split()
        text_list = [w for w in text_list if(w not in stopwords.words('spanish'))]
        #final_list = [snowball_stemmer.stem(w) for w in text_list]
        final = ' '.join(text_list)
        return final

    def load_train_data(self):
        file_train = self.file_train
        f = codecs.open(file_train, "r", "utf-8")
        sen = []
        label_list = []
        ft_train = self.file_ft_train
        ft_train = codecs.open(ft_train, "w", "utf-8")
        for line in f:
            line_list = line.split("\t")
            sen1 = line_list[0]
            sen2 = line_list[2]
            sen = sen1 + ' ' + sen2
            sen = self.clean_str(sen)
            label_list.append(line_list[4])
            label = " __label__" + str(line_list[4]).strip()
            final = sen + label +"\n"
            ft_train.write(final)
        ft_train.close()
        f.close()
        counter = Counter(label_list)
        print("counter:",counter)

    def load_test_data(self):
        file_test = self.file_test
        f = codecs.open(file_test, "r", "utf-8")
        ft_test = self.file_ft_test
        ft_test = codecs.open(ft_test, "w", "utf-8")
        for line in f:
            line_list = line.split("\t")
            sen1 = line_list[0]
            sen2 = line_list[1]
            sen = sen1 + ' ' + sen2
            sen = self.clean_str(sen)
            sen += '\n'
            ft_test.write(sen)
        ft_test.close()
        f.close()

    def process(self):
        self.load_train_data()
        self.load_test_data()


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
