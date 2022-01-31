import pandas as pd
import numpy as np
from numpy.random import seed
seed(42)
import os
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
import pickle

from Scripts.preprocess import *
from Scripts.feature_extraction import *
from Scripts.plotting_functions import *
from Scripts.data_manipulation import *
from Scripts.nn_models import baseline_model


### ELMO MODEL MOVED IN NN_MODELS SCRIPT


features = False  # If sat to true, additional extracted features will be used
model_name = "baseline"  # One of all models possible
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
tf.random.set_seed(42)
from tensorflow.keras.utils import to_categorical, plot_model

import keras
from keras.models import Model, Input
from keras.layers import Concatenate, LSTM, TimeDistributed, Dense, BatchNormalization, Bidirectional, Lambda

from keras.backend import manual_variable_initialization
manual_variable_initialization(True)

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

words = list(set(train_set["Token"].values))
n_words = len(words)
tags = list(set(train_set['BIO'].values))
n_tags = len(tags)


"""
6. Set global parameters
"""

print("set parameters")
batch_size = 32
max_len = 300

if train_mode:
    word2idx = {w: i for i, w in enumerate(words)}  # Create word-to-index-map
    word2idx_save = open("w2idx.json", "w")  # save it for further use
    json.dump(word2idx, word2idx_save)
    word2idx.close()

    tag2idx = {t: i for i, t in enumerate(tags)}  # Create tag-to-index-map
    tag2idx_save = open("t2idx.json", "w")  # save it for further use
    json.dump(tag2idx, tag2idx_save)
    tag2idx_save.close()

    idx2tag = {i: t for i, t in enumerate(tags)}  # Create index-to-tag-map
    idx2tag_save = open("i2tg.json", "w")
    json.dump(idx2tag, idx2tag_save)
    idx2tag_save.close()

else:
    # load them
    word2idx_save = open("w2idx.json", "r")
    word2idx = word2idx_save.read()
    word2idx_save.close()

    tag2idx_save = open("t2idx.json", "r")
    tag2idx = tag2idx_save.read()
    tag2idx_save.close()

    idx2tag_save = open("i2tg.json", "r")
    idx2tag = idx2tag_save.read()
    idx2tag_save.close()

num_features = 40


"""
7. Sentence preparation
"""

print("sentence preparation")
sents = group_sentences(train_set, 'BIO')
sentences = [s for s in sents if len(s) <= max_len]

if features:
    X1, X2 = prepare_and_pad(sentences, max_len)
    y = [[tag2idx[w[len(w) - 1]] for w in s] for s in sentences]
    y = pad_sequences(maxlen=max_len, sequences=y, padding="post", value=tag2idx["O"])

else:
    X = [[word2idx[w[0]] for w in s] for s in sentences]
    X = pad_sequences(maxlen=max_len, sequences=X, padding="post", value=n_words - 1)

    y = [[tag2idx[w[len(w) - 1]] for w in s] for s in sentences]
    y = pad_sequences(maxlen=max_len, sequences=y, padding="post", value=tag2idx["O"])
    y = [to_categorical(i, num_classes=n_tags) for i in y]


"""
8. Split to train and validation data
"""

print("train test split")
if features:
    X1_train, X1_valid, y_train, y_valid = train_test_split(X1, y, test_size=0.2, random_state=2021)
    X2_train, X2_valid, y_train, y_valid = train_test_split(X2, y, test_size=0.2, random_state=2021)
    X1_train = X1_train[:(len(X1_train) // batch_size) * batch_size]
    X2_train = X2_train[:(len(X2_train) // batch_size) * batch_size]
    X1_valid = X1_valid[:(len(X1_valid) // batch_size) * batch_size]
    X2_valid = X2_valid[:(len(X2_valid) // batch_size) * batch_size]

    y_train = y_train[:(len(y_train) // batch_size) * batch_size]
    y_valid = y_valid[:(len(y_valid) // batch_size) * batch_size]
    y_train = y_train.reshape(y_train.shape[0], y_train.shape[1], 1)
    y_valid = y_valid.reshape(y_valid.shape[0], y_valid.shape[1], 1)

else:
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, shuffle=False)

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
10. Build the model
"""

print("build model")
#model = baseline_model(max_len, n_words, n_tags)
#model.compile(optimizer="rmsprop", loss="categorical_crossentropy", metrics=["categorical_accuracy"])
#model.summary()


#model = build_model(max_len, n_tags)
#model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
#model.summary()

"""
11. Fit the model
"""

#model.fit(X_train, np.array(y_train), batch_size=32, epochs=10, validation_split=0.2, verbose=1)
#history = model.fit(X_train, np.array(y_train), batch_size=32, epochs=15, validation_split=0.2, verbose=1)
#hist = pd.DataFrame(history.history)

# Save model architecture in json format
#model_json = model.to_json()
#with open("baseline.json", "w") as json_file:
#    json_file.write(model_json)

#model.save_weights("model1.h5")
#model.save("new_weights.h5")

#history = model.fit([np.array(X1_train), np.array(X2_train).reshape((len(X2_train), max_len, 40))],
#                    y_train,
#                    validation_data=([np.array(X1_valid), np.array(X2_valid).reshape((len(X2_valid), max_len, 40))], y_valid),
#                    batch_size=batch_size, epochs=2, verbose=1)
#hist = pd.DataFrame(history.history)


"""
13. Load saved model
"""

# load json and create model
json_file = open('baseline.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = tf.keras.models.model_from_json(loaded_model_json)
loaded_model.compile(optimizer="rmsprop", loss="categorical_crossentropy", metrics=["categorical_accuracy"])
loaded_model.summary()
loaded_model.load_weights("model1.h5")


"""
14. Test the model with the test set
"""

p = loaded_model.predict(np.array(X_test))
p = np.argmax(p, axis=-1)
y_test = np.array(y_test)
y_test = np.argmax(y_test, axis=-1)

y_orig = []
for sent in y_test:
    for tag in sent:
        y_orig.append(tag)

y_preds = []
for sent in p:
    for tag in sent:
        y_preds.append(tag)

report = classification_report(y_orig, y_preds)
print(report)

#sents_test = group_sentences(test_set, "BIO")
#sentences_test = [s for s in sents_test if len(s) < max_len]
#X1_test, X2_test = prepare_and_pad(sentences_test, max_len)
#y_test = [[tag2idx[w[len(w) - 1]] for w in s] for s in sentences_test]
#y_test = pad_sequences(maxlen=max_len, sequences=y_test, padding="post", value=tag2idx["O"])
## If batch size is not divisible with number of samples, batch size should be redefined
#y_pred = model.predict([X1_test, np.array(X2_test).reshape((len(X2_test), max_len, 40))])

#p = np.argmax(y_pred, axis=-1)
#y_orig = []
#for sent in y_test:
#    for tag in sent:
#        y_orig.append(tag)
#y_preds = []
#for sent in p:
#    for tag in sent:
#        y_preds.append(tag)
#report = classification_report(y_orig, y_preds)
#print(report)

"""
12. Plotting learning curves
"""

#plot_learning_curves(hist, "accuracy", "val_accuracy")
#plot_learning_curves(hist, "loss", "val_loss")

