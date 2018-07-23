#!/usr/bin/env python
# -*- coding: utf-8 -*-

import nltk
import re
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
import codecs
from collections import Counter
import yaml
import numpy as np
np.random.seed(0)


class preprocess(object):
    def __init__(self, conf):
        self.file_train1 = conf["file_train1"]
        self.file_train2 = conf["file_train2"]
        self.file_test = conf["file_test"]
        self.file_ft_train = conf["file_ft_train"]
        self.file_ft_test = conf["file_ft_test"]
        self.file_ft_val = conf["file_ft_val"]

    def clean_str(self, text):
        # 词干还原工具
        # snowball_stemmer = SnowballStemmer("spanish")
        text = re.sub("[¡\!\?\¿.,:\"Ü\-\(\)\[\]\{\}\_\/\\@#$%&——]", "", text)
        text = re.sub("[(\d)*]", "", text)
        text = text.lower()
        text_list = text.split()
        text_list = [w for w in text_list if (w not in stopwords.words('spanish'))]
        # final_list = [snowball_stemmer.stem(w) for w in text_list]
        final = ' '.join(text_list)
        return final

    def load_train_data1(self):
        file_train = self.file_train1
        f = codecs.open(file_train, "r", "utf-8")
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
            final = sen + label + "\n"
            ft_train.write(final)
        ft_train.close()
        f.close()
        counter = Counter(label_list)
        print("counter:", counter)

    def load_train_data2(self):
        file_train = self.file_train2
        f = codecs.open(file_train, "r", "utf-8")
        sen = []
        label_list = []
        ft_train = self.file_ft_train
        ft_train = codecs.open(ft_train, "a", "utf-8")
        for line in f:
            line_list = line.split("\t")
            sen1 = line_list[1]
            sen2 = line_list[3]
            sen = sen1 + ' ' + sen2
            sen = self.clean_str(sen)
            label_list.append(line_list[4])
            label = " __label__" + str(line_list[4]).strip()
            final = sen + label + "\n"
            ft_train.write(final)
        ft_train.close()
        f.close()
        counter = Counter(label_list)
        print("counter:", counter)

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

    def load_validatin_data(self):
        file_train = self.file_ft_train

    def process(self):
        self.load_train_data1()
        self.load_train_data2()
        self.load_test_data()


if __name__ == '__main__':
    file_config = open("../../conf/v1.0/config.yaml")
    conf = yaml.load(file_config)
    prepro = preprocess(conf)
    prepro.process()


"""
def make_trie(words):
    trie = {}
    for word in words:
        t = trie
        for c in word:
            if c not in t: t[c] = {}
            t = t[c]
        t[None] = None
    return trie

def check_fuzzy(trie, word, path='', tol=1):
    if tol < 0:
        return set()
    elif word == '':
        results = set()
        if None in trie:
            results.add(path)
        # 增加词尾字母
        for k in trie:
            if k is not None:
                results |= check_fuzzy(trie[k], '', path+k, tol-1)
        return results
    else:
        results = set()
        # 首字母匹配
        if word[0] in trie:
            results |= check_fuzzy(trie[word[0]], word[1:], path + word[0], tol)
        # 分情形继续搜索（相当于保留待探索的回溯分支）
        for k in trie:
            if k is not None and k != word[0]:
                # 用可能正确的字母置换首字母
                results |= check_fuzzy(trie[k], word[1:], path+k, tol-1)
                # 插入可能正确的字母作为首字母
                results |= check_fuzzy(trie[k], word, path+k, tol-1)
        # 跳过余词首字母
        results |= check_fuzzy(trie, word[1:], path, tol-1)
        # 交换原词头两个字母
        if len(word) > 1:
            results |= check_fuzzy(trie, word[1]+word[0]+word[2:], path, tol-1)
        return results

'''构造trie树'''
allwords = []
for i in nlp.vocab.__iter__():
    allwords.append(i.text)
trie = make_trie(allwords)

def textpre_cor(text):
    text = textpre_punction(text.lower()).split()
    cors = []
    for word in text:
        checks = check_fuzzy(trie, word)
        if len(checks) == 0:
            continue
        if word in checks:
            cors.append(word)
        else:
            cors.append(list(checks)[0])
    return " ".join(cors)

def get_cors(word):
    return check_fuzzy(trie, word)

def make_trie(words):
    trie = {}
    for word in words:
        t = trie
        for c in word:
            if c not in t: t[c] = {}
            t = t[c]
        t[None] = None
    return trie

def check_fuzzy(trie, word, path='', tol=1):
    if tol < 0:
        return set()
    elif word == '':
        results = set()
        if None in trie:
            results.add(path)
        # 增加词尾字母
        for k in trie:
            if k is not None:
                results |= check_fuzzy(trie[k], '', path+k, tol-1)
        return results
    else:
        results = set()
        # 首字母匹配
        if word[0] in trie:
            results |= check_fuzzy(trie[word[0]], word[1:], path + word[0], tol)
        # 分情形继续搜索（相当于保留待探索的回溯分支）
        for k in trie:
            if k is not None and k != word[0]:
                # 用可能正确的字母置换首字母
                results |= check_fuzzy(trie[k], word[1:], path+k, tol-1)
                # 插入可能正确的字母作为首字母
                results |= check_fuzzy(trie[k], word, path+k, tol-1)
        # 跳过余词首字母
        results |= check_fuzzy(trie, word[1:], path, tol-1)
        # 交换原词头两个字母
        if len(word) > 1:
            results |= check_fuzzy(trie, word[1]+word[0]+word[2:], path, tol-1)
        return results



def textpre_cor(text):
    text = textpre_punction(text.lower()).split()
    cors = []
    for word in text:
        checks = check_fuzzy(trie, word)
        if len(checks) == 0:
            continue
        if word in checks:
            cors.append(word)
        else:
            cors.append(list(checks)[0])
    return " ".join(cors)
"""
