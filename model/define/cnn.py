#!/usr/bin/env python
# -*- coding: utf-8 -*-


from keras.layers import Input, Dense, Embedding, Conv2D, MaxPool2D
from keras.layers import Reshape, Flatten, Dropout, Concatenate

from keras.optimizers import Adam
from keras.models import Model
from gensim import models
import yaml
import numpy as np

def transform_data(X):
    return X

def cnnmodel(conf):
    file_w2vmodel = conf["w2vmodel"]
    word2vec = models.Word2Vec.load(file_w2vmodel)
    sequence_length = 40  # x.shape[1] # 40
    vocabulary_size = word2vec.syn0_lockf.shape[0]  # len(vocabulary_inv)
    embedding_dim = 100
    filter_sizes = [3, 4, 5]
    num_filters = 512
    drop = 0.5
    # this returns a tensor
    print("Creating Model...")
    inputs = Input(shape=(sequence_length,), dtype='int32')
    #        embedding = Embedding(input_dim=vocabulary_size, output_dim=embedding_dim, input_length=sequence_length)(inputs)
    embeddings = np.zeros((vocabulary_size + 1, embedding_dim), dtype="float32")
    for w in word2vec.wv.vocab:
        embeddings[word2vec.wv.vocab[w].index] = word2vec.wv[w]
    embedder = Embedding(vocabulary_size + 1, output_dim=embedding_dim, input_length=sequence_length, weights=[embeddings], trainable=True)(inputs)
    reshape = Reshape((sequence_length, embedding_dim, 1))(embedder)
    conv_0 = Conv2D(num_filters, kernel_size=(filter_sizes[0], embedding_dim), padding='valid',
                    kernel_initializer='normal', activation='relu')(reshape)
    conv_1 = Conv2D(num_filters, kernel_size=(filter_sizes[1], embedding_dim), padding='valid',
                    kernel_initializer='normal', activation='relu')(reshape)
    conv_2 = Conv2D(num_filters, kernel_size=(filter_sizes[2], embedding_dim), padding='valid',
                    kernel_initializer='normal', activation='relu')(reshape)
    # padding='valid'代表只对数据进行有效的卷积，对边界数据不处理
    maxpool_0 = MaxPool2D(pool_size=(sequence_length - filter_sizes[0] + 1, 1), strides=(1, 1), padding='valid')(
        conv_0)
    maxpool_1 = MaxPool2D(pool_size=(sequence_length - filter_sizes[1] + 1, 1), strides=(1, 1), padding='valid')(
        conv_1)
    maxpool_2 = MaxPool2D(pool_size=(sequence_length - filter_sizes[2] + 1, 1), strides=(1, 1), padding='valid')(
        conv_2)
    concatenated_tensor = Concatenate(axis=1)([maxpool_0, maxpool_1, maxpool_2])
    flatten = Flatten()(concatenated_tensor)
    dropout = Dropout(drop)(flatten)
    output = Dense(units=2, activation='softmax')(dropout)
    # this creates a model that includes
    model = Model(inputs=inputs, outputs=output)
    adam = Adam(lr=1e-4, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
    model.compile(optimizer=adam, loss='binary_crossentropy', metrics=['accuracy'])
    return model