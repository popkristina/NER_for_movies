# Imports
import pandas as pd
import numpy as np
import os
import json

# import tensorflow.compat.v1 as tf
# import tensorflow_hub as hub
# from tensorflow.keras.utils import to_categorical, plot_model
# from tensorflow.compat.v1.keras import backend as K

# from keras.models import Model, Input
# from keras.layers import Concatenate, LSTM, TimeDistributed, Dense, \
#     BatchNormalization, Bidirectional, Lambda

from Scripts.preprocess import *
from Scripts.feature_extraction import *
from Scripts.plotting_functions import *
# from Scripts.data_manipulation import *
from Scripts.evaluate import *
# from Scripts.train_models import *
from Scripts.assemble import *

# Insert text(s)
# TODO: Fix input type

"""texts_1 = pd.read_csv(
    "data/different_formats/MovieSuggestions_submissions_pretty.txt",
    sep='\t',
    lineterminator='\n')
texts_1['id'] = texts_1['link'].str.split("/")
texts_1['id'] = texts_1['id'].apply(lambda x: x[4])
texts_1 = texts_1[['id', 'title', 'selftext']]
"""

texts_2 = pd.read_json(
    "data/different_formats/submissions.json")
texts_2 = texts_2[['id', 'title', 'selftext']]

#texts = texts_1
texts = texts_2
texts = texts[:20]

# Preprocess text
texts['text'] = texts['title'] + " " + texts['selftext']
texts = texts[['id', 'text']]
texts['text'] = texts['text'].apply(replace)

# Tokenize text
# TODO: Speed up
tokenized_texts = pd.DataFrame()
for text, id in zip(texts.text, texts.id):
    tokenized_texts = pd.concat(
        [tokenized_texts, tokenizer_sentence(text, id)], ignore_index=False)

# Extract features from text(s)
# TODO: Removed from inference pipeline -> results unsatisfactory

# Annotate text(s)


# Assemble predictions from text(s)


