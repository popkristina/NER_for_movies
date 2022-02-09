from keras.models import Model, Input
from keras.layers import Dropout, SpatialDropout1D, Concatenate, LSTM, Embedding, TimeDistributed, Dense, BatchNormalization, Bidirectional, Lambda
import tensorflow.compat.v1 as tf
#import tensorflow as tf


def baseline_model(max_len, n_words, n_tags):
    """
    The model is built with a set of tokenized sentences
    of length 'max_len'.
    The Embedding layer creates features out of the tokens.
    """

    input = Input(shape=(max_len,))
    model = Embedding(input_dim=n_words, output_dim=max_len, input_length=max_len)(input)
    model = Dropout(0.5)(model)
    model = Bidirectional(LSTM(units=200, return_sequences=True, recurrent_dropout=0.7))(model)
    out = Dense(n_tags, activation="softmax")(model)
    return Model(input, out)


def features_model(max_len, n_words, n_tags, num_feats):
    """
    Extension of 'baseline_model'.
    Takes two inputs:
     1) Tokenized sentences and
     2) 'num_feats' token-wise features per sentence
    and concatenates them before passing them through
    output layer.
    """
    input_tokens = Input(shape=(max_len,))
    emb = Embedding(input_dim=n_words, output_dim=300, input_length=max_len)(input_tokens)

    input_feats = Input(shape=(max_len, num_feats,))
    fts = Dense(num_feats, activation='softmax')(input_feats)

    model = Concatenate()([emb, fts])
    model = Dropout(0.1)(model)
    model = Bidirectional(LSTM(units=100, return_sequences=True, recurrent_dropout=0.3))(model)

    out = TimeDistributed(Dense(n_tags, activation="softmax"))(model)
    return Model([input_tokens, input_feats], out)

def ElmoEmbedding(x):
    return elmo_model(inputs={"tokens": tf.squeeze(tf.cast(x, tf.string)),
                              "sequence_len": tf.constant(batch_size*[max_len])},
                      signature="tokens",
                      as_dict=True)["elmo"]


def elmo_model_features(max_len, n_tags):
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


def elmo_model(max_len, n_tags):
    elmo_input_layer = Input(shape=(max_len,), dtype=tf.string)
    elmo_output_layer = Lambda(ElmoEmbedding, output_shape=(None, 1024))(elmo_input_layer)
    output_layer = BatchNormalization()(elmo_output_layer)
    output_layer = Bidirectional(LSTM(units=512, return_sequences=True, recurrent_dropout=0.2, dropout=0.2))(
        output_layer)
    output_layer = TimeDistributed(Dense(n_tags, activation='softmax'))(output_layer)
    model = Model(elmo_input_layer, output_layer)
    return model