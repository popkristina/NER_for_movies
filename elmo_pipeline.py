import pandas as pd
import numpy as np
import os
import json
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
import argparse

# The main pipeline uses ELMO, which currently ony works with tf1
import tensorflow.compat.v1 as tf
import tensorflow_hub as hub
from tensorflow.keras.utils import to_categorical, plot_model
from tensorflow.compat.v1.keras import backend as K

from keras.models import Model, Input
from keras.layers import Concatenate, LSTM, TimeDistributed, Dense, \
    BatchNormalization, Bidirectional, Lambda

from Scripts.preprocess import *
from Scripts.feature_extraction import *
from Scripts.plotting_functions import *
from Scripts.data_manipulation import *
from Scripts.evaluate import *
from Scripts.train_models import *
from Scripts.assemble import *

# DEFINE GLOBAL PARAMETERS
param_dict = dict()
param_dict["batch_size"] = 32
param_dict["epochs"] = 20
param_dict["max_len"] = 300
param_dict["validation_split"] = 0.2
param_dict["num_features"] = 50
param_dict["optimizer"] = "adam"
param_dict["loss"] = "sparse_categorical_crossentropy"
param_dict["metrics"] = ["accuracy"]

features = True  # If sat to true, additional extracted features will be used
new_dicts = False
model_name = "elmo_best"


def ElmoEmbedding(x):
    return elmo_model(inputs={"tokens": tf.squeeze(tf.cast(x, tf.string)),
                              "sequence_len": tf.constant(param_dict["batch_size"]*[param_dict["max_len"]])},
                      signature="tokens",
                      as_dict=True)["elmo"]


def build_elmo_model(max_len, n_tags):
    elmo_input_layer = Input(shape=(max_len,), dtype=tf.string)
    elmo_output_layer = Lambda(ElmoEmbedding, output_shape=(None, 1024))(elmo_input_layer)
    output_layer = BatchNormalization()(elmo_output_layer)
    output_layer = Bidirectional(LSTM(units=512, return_sequences=True, recurrent_dropout=0.2, dropout=0.2))(
        output_layer)
    output_layer = TimeDistributed(Dense(n_tags, activation='softmax'))(output_layer)
    model = Model(elmo_input_layer, output_layer)
    return model


def build_model(max_len, n_tags):
    # Input Layers
    word_input_layer = Input(shape=(max_len, 50))
    elmo_input_layer = Input(shape=(max_len,), dtype=tf.string)

    word_output_layer = Dense(n_tags, activation='softmax')(word_input_layer)
    elmo_output_layer = Lambda(ElmoEmbedding, output_shape=(None, 1024))(elmo_input_layer)

    output_layer = Concatenate()([word_output_layer, elmo_output_layer])
    output_layer = BatchNormalization()(output_layer)
    output_layer = Bidirectional(LSTM(units=512, return_sequences=True, recurrent_dropout=0.2, dropout=0.2))(
        output_layer)
    output_layer = Dense(n_tags, activation='softmax')(output_layer)

    model = Model([elmo_input_layer, word_input_layer], output_layer)
    return model


"""
############ 1 READ PREPROCESSED TRAIN AND TEST DATA ################## 
"""
all_data, train_set, test_set = read_preprocessed_data('/home/kpopova/reddit_data_2022_04/')


"""
############ 2 CREATE SETS OF WORDS AND TAGS  ################## 
"""
words, n_words, tags, n_tags = create_word_and_tag_list(all_data)

if new_dicts:
    word2idx = create_dict(words)  # Create word-to-index-map
    tag2idx = create_dict(tags)  # Create tag-to-index-map
    idx2tag = {v: k for k, v in tag2idx.items()}

    save_as_json(word2idx, "w2idx")
    save_as_json(tag2idx, "t2idx")
    save_as_json(idx2tag, "i2tg")

else:
    with open("helper_dicts/w2idx.json") as word2idx_save:
        word2idx = json.load(word2idx_save)
    with open("helper_dicts/t2idx.json") as tag2idx_save:
        tag2idx = json.load(tag2idx_save)
    with open("helper_dicts/i2tg.json") as idx2tag_save:
        idx2tag = json.load(idx2tag_save)


"""
############ 3 SENTENCE PREPARATION  ################## 
"""
sentences = group_sentences(train_set, 'Sent_id', 'BIO')
for i in range(0, len(sentences)):
    sentences[i] = sentences[i][0:300]

y = [[tag2idx[w[len(w)-1]] for w in s] for s in sentences]
y = pad_sequences(maxlen=param_dict["max_len"], sequences=y, padding="post", value=tag2idx["O"])

X1 = pad_textual_data(sentences, param_dict["max_len"])
if features:
    X2 = pad_feature_data(sentences, param_dict["max_len"], param_dict["num_features"])


"""
############ 4 SPLIT TO TRAIN AND VALIDATION DATA  ################## 
"""
if features:
    X1_train, X1_valid, X2_train, X2_valid, y_train, y_valid = split_to_fit_batch(X1, y, param_dict["batch_size"], X2)
else:
    X1_train, X1_valid, y_train, y_valid = split_to_fit_batch(X1, y, param_dict["batch_size"])


"""
############ 5 SETUP KERAS SESSION PARAMETERS ################## 
"""
tf.disable_eager_execution()
elmo_model = hub.Module("https://tfhub.dev/google/elmo/3", trainable=True)
sess = tf.Session()
K.set_session(sess)
sess.run([tf.global_variables_initializer(), tf.tables_initializer()])


"""
############ 6 BUILD AND FIT THE MODEL OR LOAD IF PREV. SAVED ################## 
"""
if features:
    model = build_model(param_dict["max_len"], n_tags)
else:
    model = build_elmo_model(param_dict["max_len"], n_tags)
model.compile(optimizer=param_dict["optimizer"], loss=param_dict["loss"], metrics=[param_dict["metrics"]])
model.summary()

print("Fit model")
if features:
    model, history = train_with_features(X1_train, X2_train, X1_valid, X2_valid, y_train, y_valid, param_dict, model)
else:
    model, history = train(X1_train, X1_valid, y_train, y_valid, param_dict, model)

print("Saving model")
model_json = model.to_json()
with open("models/" + "new_elmo" + ".json", "w") as json_file:
    json_file.write(model_json)
model.save_weights("models/" + 'new_elmo' + ".h5")


    

