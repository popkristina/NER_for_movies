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

features = False  # If sat to true, additional extracted features will be used
new_dicts = False
model_name = "elmo_best"
train_mode = False
tokenized = True
input_data_format = "csv"


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


"""  ##########################   1 READ PREPROCESSED TRAIN AND TEST DATA      #######################  """

all_data, train_set, test_set = read_preprocessed_data('/home/kpopova/data/')

"""
if input_data_format == 'csv':
    test_set = pd.read_csv("submissions.csv")  # Reads data in original sentence format
    test_submissions = [submission['title'] + ' ' + submission['selftext'] for submission in test_set]
elif input_data_format == "json":
    f = open(data_path + 'submissions.json')
    test_set = json.load(f)
    test_submissions = [submission['title'] + ' ' + submission['selftext'] for submission in test_set]
    f.close()
elif input_data_format == "txt":
    test_set = pd.read_fwf(data_path + 'MovieSuggestions_submissions_pretty.txt') # Finish reading here
"""

""" #############################    2 CREATE SETS OF WORDS AND TAGS   ################################  """

words, n_words, tags, n_tags = create_word_and_tag_list(all_data)

if new_dicts:
    word2idx = create_dict(words)  # Create word-to-index-map
    tag2idx = create_dict(tags)  # Create tag-to-index-map
    idx2tag = {v: k for k, v in tag2idx.items()}

    save_as_json(word2idx, "w2idx")
    save_as_json(tag2idx, "t2idx")
    save_as_json(idx2tag, "i2tg")

else:
    with open("data/helper_dicts/w2idx.json") as word2idx_save:
        word2idx = json.load(word2idx_save)
    with open("data/helper_dicts/t2idx.json") as tag2idx_save:
        tag2idx = json.load(tag2idx_save)
    with open("data/helper_dicts/i2tg.json") as idx2tag_save:
        idx2tag = json.load(idx2tag_save)


""" ######################################          3 SENTENCE PREPARATION         ################################ """

if train_mode:
    sentences = group_sentences(train_set, 'Sent_id', 'BIO')
    y = [[tag2idx[w[len(w)-1]] for w in s] for s in sentences]
    y = pad_sequences(maxlen=param_dict["max_len"], sequences=y, padding="post", value=tag2idx["O"])

    X1 = pad_textual_data(sentences, param_dict["max_len"])
    if features:
        X2 = pad_feature_data(sentences, param_dict["max_len"], param_dict["num_features"])


""" ###############################     4 SPLIT TO TRAIN AND VALIDATION DATA      #########################  """

if train_mode:
    if features:
        X1_train, X1_valid, X2_train, X2_valid, y_train, y_valid = split_to_fit_batch(X1, y, param_dict["batch_size"], X2)
    else:
        X1_train, X1_valid, y_train, y_valid = split_to_fit_batch(X1, y, param_dict["batch_size"])

""" ###################################       5 SETUP KERAS SESSION PARAMETERS         ############################ """

tf.disable_eager_execution()
elmo_model = hub.Module("https://tfhub.dev/google/elmo/3", trainable=True)
sess = tf.Session()
K.set_session(sess)
sess.run([tf.global_variables_initializer(), tf.tables_initializer()])


""" ############################    6 BUILD AND FIT THE MODEL OR LOAD IF PREV. SAVED      #######################"""

if train_mode:
    if features:
        model = build_model(param_dict["max_len"], n_tags)
    else:
        model = build_elmo_model(param_dict["max_len"], n_tags)
    model.compile(optimizer=param_dict["optimizer"], loss=param_dict["loss"], metrics=[param_dict["metrics"]])
    model.summary()

    if features:
        model, history = train_with_features(X1_train, X2_train, X1_valid, X2_valid, y_train, y_valid, param_dict, model)
    else:
        model, history = train(X1_train, X1_valid, y_train, y_valid, param_dict, model)

    model_json = model.to_json()
    with open("models/" + model_name + ".json", "w") as json_file:
        json_file.write(model_json)
    model.save_weights("models/" + model_name + ".h5")

else:
    json_file = open("models/" + model_name + ".json", "r")
    loaded_model_json = json_file.read()
    json_file.close()
    model = tf.keras.models.model_from_json(loaded_model_json)
    model.compile(optimizer=param_dict["optimizer"], loss=param_dict["loss"], metrics=param_dict["metrics"])
    model.summary()
    model.load_weights("models/" + model_name + ".h5")


""" ###################################      7 TEST THE MODEL      #######################################"""

# If test data is not in preprocessed & tokenized form, do that first
if not tokenized:
    for i in range(0, len(test_submissions)):
        test_submissions[i] = replace(test_submissions[i])
    tokenized_texts = tokenizer(test_submissions)  # Tokenized text is in dataframe format
    test_sentences, labels, sent_ids = group_sents(tokenized_texts)  # Get a list of list of tokens, labels and ids

    if features:
        feats = extract_all_feats(test_sentences, sent_ids)
    test_set = feats  # TODO: Add additional step of labelling them if applicable, and do their grouping

# Otherwise just group the tokenized sentences according to their id for fast retrieval.
sentences_test = group_sentences(test_set, "Sent_id", "BIO")
orig_len = len(sentences_test)
""" 
There are two ways to resolve the sentence outliers issues: 
    1. We can take out those longer than a maximal length (300), 
    2. Or we can shorten the longer sentences to a maxumal length (300) """

# sentences_test = [s for s in sents_test if len(s) <= param_dict["max_len"]]
for i in range(0, len(sentences_test)):
    sentences_test[i] = sentences_test[i][0:300]
while len(sentences_test) % 32 != 0:
    sentences_test.append(sentences_test[-1])

"""Create the target variables and pad them"""
y_test = [[tag2idx[w[len(w) - 1]] for w in s] for s in sentences_test]
y_test = pad_sequences(maxlen=param_dict['max_len'], sequences=y_test, padding="post", value=tag2idx["O"])

X1_test = pad_textual_data(sentences_test, param_dict["max_len"])
if features:
    X2_test = pad_feature_data(sentences_test, param_dict["max_len"], param_dict["num_features"])

# If batch size is not divisible with number of samples, batch size should be redefined
if features:
    y_pred = model.predict([X1_test, np.array(X2_test).reshape((len(X2_test), param_dict["max_len"], param_dict["num_features"]))])
else:
    y_pred = model.predict(np.array(X1_test))

""" Flatten predictions so that they are comparable between each other """
p = np.argmax(y_pred, axis=-1)
y_orig = flatten_predictions(y_test)
y_preds = flatten_predictions(p)
report = classification_report(y_orig, y_preds)
print(report)


""" ################################       8 CREATE PREDICTIONS      #############################################"""

# Take out the additional data we added to fit the batch size
predictions = from_num_to_class(p, idx2tag)[:orig_len]
X1_test = X1_test[:orig_len]
if features:
    X2_test = X2_test[:orig_len]
sentences_test = sentences_test[:orig_len]

all_outputs, all_outputs_per_sentence = assemble_predictions(predictions, X1_test, sentences_test, param_dict["max_len"])
all_outputs_per_sentence_alt = split_keyphrases(all_outputs_per_sentence)

with open("predictions/" + model_name + "_unmatched_format_1.json", "w") as outfile:
    json.dump(all_outputs_per_sentence, outfile)

with open("predictions/" + model_name + "_unmatched_alt_format_1.json", "w") as outfile:
    json.dump(all_outputs_per_sentence_alt, outfile)
