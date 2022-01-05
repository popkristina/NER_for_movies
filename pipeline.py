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

sentences, labels, sent_ids = group_sents(tokenized_text)

"""
3. Extract features from text
"""

feats1 = spacy_feats_all(sentences, sent_ids)
feats2 = spacy_feats_tensors_all(sentences)
feats3 = sentiment_feats_all(sentences, feats1[["Sentence", "Token"]])
feats4 = tf_feats(sentences, feats1[["Sentence", "Token"]])
feats = pd.concat([feats1, feats2, feats3, feats4], axis=1)


"""
4. Read preprocessed train and test data
"""

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

batch_size = 32
#batch_size = 2
max_len = 300
train_mode = True
tag2idx = {t: i for i, t in enumerate(tags)}
idx2tag = {i: t for i, t in enumerate(tags)}
num_features = 40

"""
7. Sentence preparation
"""

sents = group_sentences(train_set, 'BIO')
sentences = [s for s in sents if len(s) <= max_len]
X1, X2 = prepare_and_pad(sentences, max_len)
y = [[tag2idx[w[len(w) - 1]] for w in s] for s in sentences]
y = pad_sequences(maxlen=max_len, sequences=y, padding="post", value=tag2idx["O"])


"""
8. Split to train and validation data
"""

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

"""
9. Setup keras session parameters
"""

tf.disable_eager_execution()
elmo_model = hub.Module("https://tfhub.dev/google/elmo/3", trainable=True)
sess = tf.Session()
K.set_session(sess)
sess.run([tf.global_variables_initializer(), tf.tables_initializer()])

# Load saved session: new_saver = tf.train.import_meta_graph('utilities/my_test_model-1000.meta')
# Load saved session: new_saver.restore(sess, tf.train.latest_checkpoint('utilities/'))

#saver = tf.train.Saver()

"""
10. Build the model
"""

model = build_model(max_len, n_tags)
model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
model.summary()

"""
11. Fit the model
"""
history = model.fit([np.array(X1_train), np.array(X2_train).reshape((len(X2_train), max_len, 40))],
                    y_train,
                    validation_data=([np.array(X1_valid), np.array(X2_valid).reshape((len(X2_valid), max_len, 40))], y_valid),
                    batch_size=batch_size, epochs=2, verbose=1)
#saver.save(sess, 'utilities/my_test_model', global_step=1000)
hist = pd.DataFrame(history.history)


"""
12. Plotting learning curves
"""

plot_learning_curves(hist, "accuracy", "val_accuracy")
plot_learning_curves(hist, "loss", "val_loss")


"""
13. Save trained model
"""

#import tensorflow as tf
#model.save("/home/kpopova/project/utilities/new_model")
#new_model = load_model('C:/Users/Kiki/Projects/ner_movies/Scripts/my_model')

"""
14. Test the model with the test set
"""

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
