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
from Scripts.nn_models import baseline_model, features_model
from Scripts.test_models import flatten_predictions
from Scripts.train_models import *


def ElmoEmbedding(x):
    return elmo_model(inputs={"tokens": tf.squeeze(tf.cast(x, tf.string)),
                              "sequence_len": tf.constant(batch_size*[max_len])},
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


features = True # If sat to true, additional extracted features will be used
train_mode = True
model_name = "elmo_best"

"""
Option 1: Input one submission
"""
input_submission = 'Please recommend me some movies like: The Fault in Our Stars. I ' \
                   'like actors like Logan Lerman and Mischa Burton. Please no horrors ' \
                   'or thrillers.'

"""
Option 2: Input dataset
"""

# TODO: Add Input dataset


# 1. PREPROCESS AND TOKENIZE INPUT TEXT
text = replace(input_submission)
texts = [text]
tokenized_text = tokenizer(texts)  # Tokenized text is in dataframe format

# 2. GROUP SENTENCES TO GET A LIST OF TOKENS, LABELS AND SENTENCE IDS
sentences, labels, sent_ids = group_sents(tokenized_text)

# 3. EXTRACT FEATURES FROM TEXT
if features:
    feats = extract_all_feats(sentences, sent_ids)


"""
4. Read preprocessed train and test data
"""

print("Read preprocessed data")
path = '/home/kpopova/project/data'
all_data, train_set, test_set = read_preprocessed_data(path)

"""
5. Create sets of words and tags (for training)
"""

words = list(set(all_data["Token"].values))
words.append('ENDPAD')
n_words = len(words)
tags = list(set(train_set['BIO'].values))
n_tags = len(tags)

"""
6. Set global parameters
"""

print("set parameters")
max_len = 300
param_dict = dict()
param_dict["batch_size"] = 32
param_dict["epochs"] = 20
param_dict["max_len"] = 300
param_dict["validation_split"] = 0.2
param_dict["num_features"] = 50
param_dict["optimizer"] = "adam"
param_dict["loss"] = "sparse_categorical_crossentropy"
param_dict["metrics"] = "accuracy"


if train_mode:
    word2idx = create_dict(words)  # Create word-to-index-map
    word2idx_save = open("helper_dicts/w2idx.json", "w")  # save it for further use
    json.dump(word2idx, word2idx_save)
    word2idx_save.close()

    tag2idx = create_dict(tags)  # Create tag-to-index-map
    tag2idx_save = open("helper_dicts/t2idx.json", "w")  # save it for further use
    json.dump(tag2idx, tag2idx_save)
    tag2idx_save.close()

    idx2tag = {v: k for k, v in tag2idx.items()}
    #idx2tag = create_dict(tags, reverse=True)  # Create index-to-tag-map
    idx2tag_save = open("helper_dicts/i2tg.json", "w")
    json.dump(idx2tag, idx2tag_save)
    idx2tag_save.close()


else:
    with open("helper_dicts/w2idx.json") as word2idx_save:
        word2idx = json.load(word2idx_save)
    with open("helper_dicts/t2idx.json") as tag2idx_save:
        tag2idx = json.load(tag2idx_save)
    with open("helper_dicts/i2tg.json") as idx2tag_save:
        idx2tag = json.load(idx2tag_save)

"""
7. Sentence preparation
"""

print("Sentence preparation")
sents = group(train_set, 'BIO')
sentences = [s for s in sents if len(s) <= max_len]

y = [[tag2idx[w[len(w)-1]] for w in s] for s in sentences]
y = pad_sequences(maxlen=max_len, sequences=y, padding="post", value=tag2idx["O"])

X1 = pad_textual_data(sentences, max_len)

if features:
    X2 = pad_feature_data(sentences, max_len, param_dict["num_features"])


"""
8. Split to train and validation data
"""
batch_size = param_dict["batch_size"]
X1_train, X1_valid, y_train, y_valid = split_to_fit_batch(X1, y, batch_size)

if features:
    X2_train, X2_valid, _, _ = train_test_split(X2, y, test_size=0.2, random_state=2021)
    X2_train = X2_train[:(len(X2_train) // batch_size) * batch_size]
    X2_valid = X2_valid[:(len(X2_valid) // batch_size) * batch_size]


"""
#9. Setup keras session parameters
"""

tf.disable_eager_execution()
elmo_model = hub.Module("https://tfhub.dev/google/elmo/3", trainable=True)
sess = tf.Session()
K.set_session(sess)
sess.run([tf.global_variables_initializer(), tf.tables_initializer()])


"""
#10. Build and fit the model or load if previously saved
"""

if train_mode:
    print("Build model")
    if features:
        model = build_model(max_len, n_tags)
    else:
        model = elmo_model(max_len, n_tags)
    model.compile(optimizer=param_dict["optimizer"], loss=param_dict["loss"], metrics=[param_dict["metrics"]])
    model.summary()

    print("Fit model")

    if features:
        model, history = train_with_features(X1_train, X2_train, X1_valid, X2_valid, y_train, y_valid, param_dict, model)

    else:
        model, history = train(X1_train, X1_valid, y_train, y_valid, param_dict, model)

    print("Saving model")
    model_json = model.to_json()
    with open("models/" + model_name + ".json", "w") as json_file:
        json_file.write(model_json)
    model.save_weights("models/" + model_name + ".h5")


else:
    # load json and create model
    json_file = open("models/" + model_name + ".json", "r")
    loaded_model_json = json_file.read()
    json_file.close()
    model = tf.keras.models.model_from_json(loaded_model_json)
    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
    model.summary()
    model.load_weights(model_name + ".h5")


"""
#14. Test the model with the test set
"""

sents_test = group(test_set, "BIO")
sentences_test = [s for s in sents_test if len(s) <= max_len]

y_test = [[tag2idx[w[len(w) - 1]] for w in s] for s in sentences_test]
y_test = pad_sequences(maxlen=max_len, sequences=y_test, padding="post", value=tag2idx["O"])

X_words_test = pad_textual_data(sentences_test, max_len)

if features:
    X_features_test = pad_feature_data(sentences_test, max_len, num_features)


## If batch size is not divisible with number of samples, batch size should be redefined
if features:
    y_pred = model.predict([X_words_test_test, np.array(X_features_test).reshape((len(X_features_test), max_len, num_features))])
else:
    y_pred = model.predict(X_words_test)

p = np.argmax(y_pred, axis=-1)
y_orig = flatten_predictions(y_test)
y_preds = flatten_predictions(p)
report = classification_report(y_orig, y_preds)
print(report)

#report = classification_report(all_true_labels, all_preds)
#print(report)

