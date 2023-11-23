# Imports
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
import pickle

import tensorflow.compat.v1 as tf
import tensorflow_hub as hub
from tensorflow.keras.utils import to_categorical, plot_model
from tensorflow.compat.v1.keras import backend as K

from keras.preprocessing.sequence import pad_sequences
from keras.models import Model, Input, load_model
from keras.layers import Concatenate, LSTM, Dense, BatchNormalization, Bidirectional, Lambda


def read_data(path):
    os.chdir(path)
    train = pd.read_csv("train_final_all.csv")
    test = pd.read_csv("test_final_all.csv")
    data = train.append(test)
    return data, train, test


def create_lists(data, category):
    words = list(set(data["Token"].values))
    n_words = len(words)
    tags = list(set(data[category].values))
    n_tags = len(tags)

    return words, n_words, tags, n_tags


def group_sentences(data, category):
    all_sents = []
    sent_ids = data['Sent_id'].unique()
    for curr_id in sent_ids:
        tmp_df = data[data['Sent_id'] == curr_id]
        tmp_df = pd.concat([tmp_df['Token'], tmp_df["Token_index"], tmp_df.iloc[:, 4:44], tmp_df[category]], axis=1)
        records = tmp_df.to_records(index=False)
        all_sents.append(records)
    return all_sents


def remove_sents_over_threshold(sents, threshold):
    sentences = list()
    for s in sents:
        if len(s) < threshold:
            sentences.append(s)
    return sentences


def prepare_and_pad(sentences, max_len, tag2idx):
    X1 = [[w[0] for w in s] for s in sentences]

    new_X = []
    for seq in X1:
        new_seq = []
        for i in range(max_len):
            try:
                new_seq.append(seq[i])
            except:
                new_seq.append("__PAD__")
        new_X.append(new_seq)
    X1 = new_X

    X2 = []
    for sentence in sentences:
        sent_ft = list()
        for word in sentence:
            ft = list()
            for i in range(1, 41):
                ft.append(word[i])
            sent_ft.append(ft)
        for j in range(len(sentence) - 1, max_len - 1):
            ft = list()
            for i in range(1, 41):
                ft.append(0)
            sent_ft.append(ft)
        X2.append(sent_ft)

    y = [[tag2idx[w[len(w) - 1]] for w in s] for s in sentences]
    y = pad_sequences(maxlen=max_len, sequences=y, padding="post", value=tag2idx["O"])
    return X1, X2, y


def ElmoEmbedding(x):
    return elmo_model(inputs={"tokens": tf.squeeze(tf.cast(x, tf.string)),
                              "sequence_len": tf.constant(batch_size*[max_len])},
                      signature="tokens",
                      as_dict=True)["elmo"]


def build_model(max_len, n_tags):
    # Input Layers
    word_input_layer = Input(shape=(max_len, 40))
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


def plot_learning_curves(hist, curve1, curve2):
    plt.figure(figsize=(6, 6))
    plt.plot(hist[curve1])
    plt.plot(hist[curve2])
    plt.show()


path = 'data'
batch_size = 32
all_data, train_set, test_set = read_data(path)

print("Creating sets of words and tags...")
words, n_words, tags, n_tags = create_lists(all_data, "BIO")

print("Creating sentence list...")
sents = group_sentences(train_set, 'BIO')

print("Removing submissions longer than threshold...")
sentences = remove_sents_over_threshold(sents, 300)

print("Creating word and tag maps...")
max_len = 300
tag2idx = {t: i for i, t in enumerate(tags)}

print("Preparing and padding training data...")
X1, X2, y = prepare_and_pad(sentences, max_len, tag2idx)

print("Splitting data...")
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

print("Setting parameters...")
tf.compat.v1.disable_eager_execution()
sess = tf.compat.v1.Session()
#sess = tf.Session()
K.set_session(sess)
elmo_model = hub.Module("https://tfhub.dev/google/elmo/3", trainable=True)
sess.run(tf.global_variables_initializer())
sess.run(tf.tables_initializer())

print("Building the model...")
model = build_model(max_len, n_tags)
model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
model.summary()

print("Fitting the model....")
history = model.fit([np.array(X1_train), np.array(X2_train).reshape((len(X2_train), max_len, 40))],
                    y_train,
                    validation_data=(
                    [np.array(X1_valid), np.array(X2_valid).reshape((len(X2_valid), max_len, 40))], y_valid),
                    batch_size=batch_size, epochs=3, verbose=1)
hist = pd.DataFrame(history.history)

print("Plotting learning curves...")
plot_learning_curves(hist, "accuracy", "val_accuracy")
plot_learning_curves(hist, "loss", "val_loss")

#os.chdir(path)
#model.save("new_model")
#new_model = load_model('C:/Users/Kiki/Projects/ner_movies/Scripts/my_model')


################# TEST ################################

print("Creating sentence list...")
sents_test = group_sentences(test_set, "BIO")
sentences_test = [s for s in sents_test if len(s) < max_len]

print("Preparing and padding training data...")
X1_test, X2_test, y_test = prepare_and_pad(sentences_test, max_len, tag2idx)

print("Make predictions...")
# y_pred = reconstructed.predict([X1_test, np.array(X2_test).reshape((len(X2_test), max_len, 40))])
y_pred = model.predict([X1_test, np.array(X2_test).reshape((len(X2_test), max_len, 40))])
p = np.argmax(y_pred, axis=-1)
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
