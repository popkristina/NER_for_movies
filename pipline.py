# Imports
import pandas as pd
import numpy as np
import os
import json

import torch
from transformers import BertTokenizer, RobertaTokenizer
# import tensorflow.compat.v1 as tf
#import tensorflow as tf
# import tensorflow_hub as hub
# from tensorflow.keras.utils import to_categorical, plot_model
# from tensorflow.compat.v1.keras import backend as K

from Scripts.preprocess import *
from transformers import BertForTokenClassification, RobertaForTokenClassification
from Scripts.data_manipulation import *
from Scripts.evaluate import *
from Scripts.assemble import *


def generate_predictions(grouped_submissions, model, tokenizer):
    all_predictions_list = []

    for sub in grouped_submissions:
        tokenized_submission = tokenizer.encode(sub)
        input_ids = torch.tensor([tokenized_submission])
        input_masks = torch.tensor([[float(i != 0.0) for i in ii] for ii in input_ids])
        with torch.no_grad():
            output = model(input_ids, attention_mask=input_masks)
        label_indices = np.argmax(output[0].to('cpu').numpy(), axis=2)
        label_indices = label_indices[0]
        label_indices = label_indices[1:-1]

        tokens = tokenizer.convert_ids_to_tokens(input_ids.to('cpu').numpy()[0])
        new_tokens, new_labels = [], []
        for token, label_idx in zip(tokens, label_indices):
            if token.startswith("##"):
                new_tokens[-1] = new_tokens[-1] + token[2:]
            else:
                new_labels.append(tag_values[label_idx])
                new_tokens.append(token)
        all_predictions_list.append(new_labels)
    return all_predictions_list


def generate_predictions_1(grouped_submissions, model):
    all_predictions_list = []



    return all_predictions_list

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

tag_values = ['B-actor-pos',
              'B-movie-neg',
              'B-gen-neg',
              'I-keyword-neg',
              'O', 'I-gen-neg',
              'I-actor-pos',
              'B-keyword-pos',
              'I-actor-neg',
              'B-gen-pos',
              'B-keyword-neg',
              'B-actor-neg',
              'I-movie-pos',
              'I-gen-pos',
              'I-keyword-pos',
              'B-movie-pos',
              'I-movie-neg',
              'PAD']


# Tokenize text (for ELMO)
# TODO: Speed up
tokenized_texts = pd.DataFrame()
for text, id in zip(texts.text, texts.id):
    tokenized_texts = pd.concat(
        [tokenized_texts, tokenizer_sentence(text, id)], ignore_index=False)

# Extract features from text(s)
# TODO: Removed from inference pipeline -> results unsatisfactory

#################################################################################################
# Load BERT models
bert_large_cased = BertForTokenClassification.from_pretrained("models/bert-large-cased/")
bert_base_mult = BertForTokenClassification.from_pretrained("models/bert-base-multilingual-cased/")
roberta = RobertaForTokenClassification.from_pretrained("models/roberta-large/")

# Load BERT tokenizers
tokenizer_bert_large = BertTokenizer.from_pretrained('bert-large-cased', do_lower_case=False)
tokenizer_bert_mult = BertTokenizer.from_pretrained('bert-base-multilingual-cased', do_lower_case=False)
tokenizer_roberta = RobertaTokenizer.from_pretrained('roberta-large', do_lower_case=False)

# Load ELMO model
json_file = open("models/elmo_best.json", "r")
loaded_model_json = json_file.read()
json_file.close()
#bilstm_elmo = tf.keras.models.model_from_json(loaded_model_json)
#model.compile(optimizer=param_dict["optimizer"], loss=param_dict["loss"], metrics=param_dict["metrics"])
#model.summary()
#model.load_weights("models/" + model_name + ".h5")

# Prepare submissions for inference
grouped = group_sents_nolabels(tokenized_texts)
grouped_submissions = [[word[1] for word in sub] for sub in grouped]

# Predictions generation
predictions_roberta = generate_predictions(
    grouped_submissions, roberta, tokenizer_roberta)
_, assembled_predictions_roberta = assemble_predictions(
    predictions_roberta, grouped_submissions, grouped)

predictions_bert_large_cased = generate_predictions(
    grouped_submissions, bert_large_cased, tokenizer_bert_large)
_, assembled_predictions_bert_lg_cased = assemble_predictions(
    predictions_bert_large_cased, grouped_submissions, grouped)

predictions_bert_mult = generate_predictions(
    grouped_submissions, bert_base_mult, tokenizer_bert_mult)
_, assembled_predictions_bert_mult = assemble_predictions(
    predictions_bert_mult, grouped_submissions, grouped)


# Assemble predictions from text(s)
ensemble_predictions = ensemble_appr(assembled_predictions_roberta, assembled_predictions_bert_lg_cased, assembled_predictions_bert_mult, [])
print(ensemble_predictions)

