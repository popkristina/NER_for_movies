import pandas as pd
import numpy as np
import os
import json
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
import argparse

import tensorflow as tf
#import tensorflow.compat.v1 as tf
import tensorflow_hub as hub
from tensorflow.keras.utils import to_categorical, plot_model
#from tensorflow.compat.v1.keras import backend as K

from keras.models import Model, Input
from keras.layers import Concatenate, LSTM, TimeDistributed, Dense, BatchNormalization, Bidirectional, Lambda


from Scripts.preprocess import *
from Scripts.feature_extraction import *
from Scripts.plotting_functions import *
from Scripts.data_manipulation import *
from Scripts.nn_models import baseline_model, features_model
from Scripts.test_models import flatten_predictions
from Scripts.bert_finetune_lib import *


parser = argparse.ArgumentParser(description='Additional arguments for selective operations.')
parser.add_argument('-m', '--model', dest='model_name', help='Specify the NN model to use', default="elmo", type=str)
parser.add_argument('-tr', '--train', dest='train_mode', help='Flag whether the script should initiate training',
                    default=False, type=bool)
parser.add_argument('-ft', '--features', dest='features', help='Flag if the model should extract features',
                    default=False, type=bool)
#parser.add_argument('-i', '--input_path', dest='input_path', help='Directory where input files are stored.',
#                    required=True)
#parser.add_argument('-o', '--output_path', dest='output_path', help='Directory to store results in.',
#                    required=True)
args = parser.parse_args()


def ElmoEmbedding(x):
    return elmo_model(inputs={"tokens": tf.squeeze(tf.cast(x, tf.string)),
                              "sequence_len": tf.constant(batch_size*[max_len])},
                      signature="tokens",
                      as_dict=True)["elmo"]


features = False  # If sat to true, additional extracted features will be used

feat_based_models = ["char", "char_fts", "baseline", "baseline_fts", "elmo", "elmo_fts"]
bert_models = ["bert_cs", "bert_unc", "funnel", "bert_mult_unc", "bert_mult_cs", "electra"]

model_name = "baseline"  # One of all models possible
if model_name in bert_models:
    bert_flag = True
train_mode = True
#if features and "bert" in model_name.lower():
#    throw_error()

tensorflow_version = 2
if "elmo" in model_name.lower():
    tensorflow_version = 1

if tensorflow_version == 1:
    import tensorflow.compat.v1 as tf
    import tensorflow_hub as hub
    from tensorflow.compat.v1.keras import backend as K
else:
    import tensorflow as tf


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

"""
1. Preprocess and tokenize input text
"""

text = replace(input_submission)
texts = [text]
tokenized_text = tokenizer(texts)  # Tokenized text is in dataframe format


"""
2. Group sentences in such a way to get separate lists for the sentences, their labels
    and their ids
"""

print("group sentences")
sentences, labels, sent_ids = group_sents(tokenized_text)

"""
3. Extract features from text
"""

print("Extract features")
feats1 = spacy_feats_all(sentences, sent_ids)
feats2 = spacy_feats_tensors_all(sentences)
feats3 = sentiment_feats_all(sentences, feats1[["Sentence", "Token"]])
feats4 = tf_feats(sentences, feats1[["Sentence", "Token"]])
feats = pd.concat([feats1, feats2, feats3, feats4], axis=1)


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
batch_size = 32
epochs = 15
max_len = 300
valid_split = 0.2
num_features = 50
optimizer = "adam"
loss = "sparse_categorical_crossentropy"
metrics = "accuracy"
learning_rate = 1e-05 #relevant for bert
max_grad_norm = 8 #relevant for bert

if train_mode:
    word2idx = create_dict(words)  # Create word-to-index-map
    word2idx_save = open("w2idx.json", "w")  # save it for further use
    json.dump(word2idx, word2idx_save)
    word2idx_save.close()

    tag2idx = create_dict(tags)  # Create tag-to-index-map
    tag2idx_save = open("t2idx.json", "w")  # save it for further use
    json.dump(tag2idx, tag2idx_save)
    tag2idx_save.close()

    idx2tag = create_dict(tags, reverse=True)  # Create index-to-tag-map
    idx2tag_save = open("i2tg.json", "w")
    json.dump(idx2tag, idx2tag_save)
    idx2tag_save.close()


else:
    with open("w2idx.json") as word2idx_save:
        word2idx = json.load(word2idx_save)
    with open("t2idx.json") as tag2idx_save:
        tag2idx = json.load(tag2idx_save)
    with open("i2tg.json") as idx2tag_save:
        idx2tag = json.load(idx2tag_save)

"""
7. Sentence preparation
"""

print("Sentence preparation")
sents = group_sentences(train_set, 'BIO')
sents = [s for s in sents if len(s) <= max_len]
if bert_flag:
    sentences = [[word[0] for word in sentence] for sentence in sents]
else:
    sentences = sents

y = [[tag2idx[w[len(w)-1]] for w in s] for s in sents]
y = pad_sequences(maxlen=max_len, sequences=y, padding="post", value=tag2idx["O"])

if "elmo" in model_name:
    X_words = pad_textual_data(sentences, max_len)
elif "baseline" in model_name:
    X_words = [[word2idx[w[0]] for w in s] for s in sentences]
    X_words = pad_sequences(maxlen=max_len, sequences=X_words, padding="post", value=n_words-1)
if features:
    X_features, X2 = pad_feature_data(sentences, max_len, num_features, word2idx)


"""
8. Split to train and validation data
"""

if 'elmo' in model_name:
    X_words_train, X_words_valid, y_train, y_valid = train_test_split(X_words, y, test_size=0.2, random_state=2021)
    X_words_train = X_words_train[:(len(X_words_train) // batch_size) * batch_size]
    X_words_valid = X_words_valid[:(len(X_words_valid) // batch_size) * batch_size]
    y_train = y_train[:(len(y_train) // batch_size) * batch_size]
    y_valid = y_valid[:(len(y_valid) // batch_size) * batch_size]
    y_train = y_train.reshape(y_train.shape[0], y_train.shape[1], 1)
    y_valid = y_valid.reshape(y_valid.shape[0], y_valid.shape[1], 1)

    if features:
        X_features_train, X_features_valid, _, _ = train_test_split(X_features, y, test_size=0.2, random_state=2021)
        X_features_train = X_features_train[:(len(X_features_train) // batch_size) * batch_size]
        X_features_valid = X_features_valid[:(len(X_features_valid) // batch_size) * batch_size]

elif features:
    X_words_train = X_words
    X_features_train = X_features
    y_train = y

else:
    X_words_train = X_words
    y_train = y


"""
9. Setup keras session parameters
"""

if "elmo" in model_name:
    tf.disable_eager_execution()
    elmo_model = hub.Module("https://tfhub.dev/google/elmo/3", trainable=True)
    sess = tf.Session()
    K.set_session(sess)
    sess.run([tf.global_variables_initializer(), tf.tables_initializer()])


"""
10. Build and fit the model or load if previously saved
"""

if train_mode:
    print("Build model")
    model = baseline_model(max_len, n_words, n_tags)
    model.compile(optimizer=optimizer, loss=loss, metrics=[metrics])
    model.summary()

    print("Fit model")
    history = model.fit(X_train, np.array(y_train), batch_size=batch_size, epochs=epochs, validation_split=valid_split, verbose=1)
    #history = model.fit([X1_train, np.array(X2_train).reshape((len(X2_train), max_len, num_features))],
    #                    np.array(y_train), batch_size=batch_size, epochs=15, validation_split=0.2, verbose=1)
    hist = pd.DataFrame(history.history)

    #history = model.fit([np.array(X1_train), np.array(X2_train).reshape((len(X2_train), max_len, 40))], y_train,
    #                    validation_data=([np.array(X1_valid), np.array(X2_valid).reshape((len(X2_valid), max_len, 40))],
    #                                     y_valid), batch_size=batch_size, epochs=15, verbose=1)

    #plot_learning_curves(hist, "accuracy", "val_accuracy")
    #plot_learning_curves(hist, "loss", "val_loss")

    print("Saving model")
    model_json = model.to_json()
    with open(model_name + ".json", "w") as json_file:
        json_file.write(model_json)
    model.save_weights(model_name + ".h5")


else:
    # load json and create model
    json_file = open(model_name + ".json", "r")
    loaded_model_json = json_file.read()
    json_file.close()
    model = tf.keras.models.model_from_json(loaded_model_json)
    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
    model.summary()
    model.load_weights(model_name + ".h5")


"""
14. Test the model with the test set
"""

sents_test = group_sentences(test_set, "BIO")
sentences_test = [s for s in sents_test if len(s) <= max_len]

y_test = [[tag2idx[w[len(w) - 1]] for w in s] for s in sentences_test]
y_test = pad_sequences(maxlen=max_len, sequences=y_test, padding="post", value=tag2idx["O"])

if 'elmo' in model_name:
    X_words_test = pad_textual_data(sentences_test, max_len)
elif 'baseline' in model_name:
    X_words_test = [[word2idx[w[0]] for w in s] for s in sentences_test]
    X_words_test = pad_sequences(maxlen=max_len, sequences=X_words_test, padding="post", value=n_words - 1)

if features:
    X_features_test, X2_test = pad_feature_data(sentences_test, max_len, num_features, word2idx)


## If batch size is not divisible with number of samples, batch size should be redefined
#y_pred = loaded_model.predict([X1_test, np.array(X2_test).reshape((len(X2_test), max_len, 40))])
#y_pred = model.predict([X1_test, np.array(X2_test).reshape((len(X2_test), max_len, num_features))])
y_pred = model.predict(X_words_test)

p = np.argmax(y_pred, axis=-1)
y_orig = flatten_predictions(y_test)
y_preds = flatten_predictions(p)
report = classification_report(y_orig, y_preds)
print(report)
