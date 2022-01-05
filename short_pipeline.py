import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
import pickle

#import tensorflow as tf
import tensorflow.compat.v1 as tf
import tensorflow_hub as hub
from tensorflow.keras.utils import to_categorical, plot_model
from tensorflow.compat.v1.keras import backend as K

from keras.models import Model, Input
from keras.layers import Concatenate, LSTM, TimeDistributed, Dense, BatchNormalization, Bidirectional, Lambda

from preprocess import *
from feature_extraction import *
from plotting_functions import *
from data_manipulation import *


def ElmoEmbedding(x):
    return elmo_model(inputs={"tokens": tf.squeeze(tf.cast(x, tf.string)),
                              "sequence_len": tf.constant(batch_size*[max_len])},
                      signature="tokens",
                      as_dict=True)["elmo"]


def build_model(max_len, n_tags):
    word_input_layer = Input(shape=(max_len, 40,))
    elmo_input_layer = Input(shape=(max_len,), dtype=tf.string)

    word_output_layer = Dense(n_tags, activation='softmax')(word_input_layer)
    elmo_output_layer = Lambda(ElmoEmbedding, output_shape=(None, 1024))(elmo_input_layer)

    output_layer = Concatenate()([word_output_layer, elmo_output_layer])
    output_layer = BatchNormalization()(output_layer)
    output_layer = Bidirectional(LSTM(units=512, return_sequences=True, recurrent_dropout=0.2, dropout=0.2))(
        output_layer)
    output_layer = TimeDistributed(Dense(n_tags, activation='softmax'))(output_layer)

    model = Model([elmo_input_layer, word_input_layer], output_layer)

    return model


"""
1. Read preprocessed train and test data
"""

path = '/home/kpopova/project/data'
all_data, train_set, test_set = read_preprocessed_data(path)


"""
2. Create sets of words and tags (for training)
"""

words = list(set(train_set["Token"].values))
n_words = len(words)
tags = list(set(train_set['BIO'].values))
n_tags = len(tags)


"""
3. Set global parameters
"""

batch_size = 32
#batch_size = 2
max_len = 300
train_mode = True
tag2idx = {t: i for i, t in enumerate(tags)}
idx2tag = {i: t for i, t in enumerate(tags)}
num_features = 40

"""
4. Sentence preparation
"""

sents = group_sentences(train_set, 'BIO')
sentences = [s for s in sents if len(s) <= max_len]
X1, X2 = prepare_and_pad(sentences, max_len)
y = [[tag2idx[w[len(w) - 1]] for w in s] for s in sentences]
y = pad_sequences(maxlen=max_len, sequences=y, padding="post", value=tag2idx["O"])
