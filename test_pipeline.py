import pandas as pd
import os
import json

from Scripts.data_manipulation import *
from Scripts.preprocess import *
from Scripts.feature_extraction import *

import tensorflow.compat.v1 as tf
import tensorflow_hub as hub
from tensorflow.keras.utils import to_categorical, plot_model
from tensorflow.compat.v1.keras import backend as K


data_path = "data/different_formats/"
model_name = "baseline_model_features"
input_data_format = "csv"  # possibilities: obj, json, txt
features = True
tokenized = True

test_1 = True
test_2 = False
test_3 = False

""" #######################################       1 READ INPUT        ############################################ """


if input_data_format == "csv":
    """ These are preprocessed, in practice they won't be """
    if tokenized:
        test_set = pd.read_csv(data_path + "test_final_all.csv")  # Reads the preprocessed and per-token labelled data
    else:
        test_set = pd.read_csv("submissions.csv")  # Reads data in original sentence format
        test_submissions = [submission['title'] + ' ' + submission['selftext'] for submission in test_set]

elif input_data_format == "json":
    """For now we only have preprocessed data in .csv format, so json implies the data isn't tokenized and labeled"""
    f = open(data_path + 'submissions.json')
    test_set = json.load(f)
    test_submissions = [submission['title'] + ' ' + submission['selftext'] for submission in test_set]
    f.close()

elif input_data_format == "txt":
    """Same for txt format, implies no preprocessing was done so we have to do it"""
    f = open(data_path + 'MovieSuggestions_submissions_pretty.txt')
    # TODO: Handle the reading of this type of file


""" #####################    2 PREPROCESS, TOKENIZE, GROUP INPUT TEXT AND EXTRACT FEATURES  ####################### """

# If data is not in preprocessed & tokenized form, do that first
if not tokenized:
    for i in range(0, len(test_submissions)):
        test_submissions[i] = replace(test_submissions[i])
    tokenized_texts = tokenizer(test_submissions)  # Tokenized text is in dataframe format
    test_sentences, labels, sent_ids = group_sents(tokenized_texts)  # Get a list of list of tokens, labels and ids

    if features:
        feats = extract_all_feats(test_sentences, sent_ids)

# Otherwise just group the tokenized sentences according to their id for fast retrieval.
else:
    sentences_test = group_sentences(test_set, "Sent_id", "BIO")
    for i in range(0, len(sentences_test)):
        sentences_test[i] = sentences_test[i][0:300]

    """ 
    There are two ways to resolve the sentence outliers issues: 
        1. We can take out those longer than a maximal length (300), 
        2. Or we can shorten the longer sentences to a maxumal length (300) """
    # sentences_test = [s for s in sents_test if len(s) <= param_dict["max_len"]]
    sentences_test.append(sentences_test[-1])


""" #####################################        3 LOAD A TRAINED MODEL        #################################### """

if test_1 or test_2:

    # load json and create model
    json_file = open("models/" + model_name + ".json", "r")
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = tf.keras.models.model_from_json(loaded_model_json)
    loaded_model.load_weights("models/" + model_name + ".h5")

    #batch_size = 2
    #if features:
    #    model = build_model(param_dict["max_len"], n_tags)
    #else:
    #    model = build_elmo_model(param_dict["max_len"], n_tags)

    #model.compile(optimizer=param_dict["optimizer"], loss=param_dict["loss"], metrics=param_dict["metrics"])
    #model.load_weights("models/" + model_name + ".h5")

"""

"""
############ 7 TEST THE MODEL WITH THE TEST SET ################## 
"""


y_test = [[tag2idx[w[len(w) - 1]] for w in s] for s in sentences_test]
y_test = pad_sequences(maxlen=param_dict["max_len"], sequences=y_test, padding="post", value=tag2idx["O"])
np.append(y_test, y_test[-1])
print(len(y_test))

X1_test = pad_textual_data(sentences_test, param_dict["max_len"])
if features:
    X2_test = pad_feature_data(sentences_test, param_dict["max_len"], param_dict["num_features"])


# If batch size is not divisible with number of samples, batch size should be redefined
if features:
    y_pred = model.predict(
        [X1_test, np.array(X2_test).reshape((len(X2_test), param_dict["max_len"], param_dict["num_features"]))], batch_size=2)
else:
    y_pred = model.predict(X1_test, batch_size=2)

y_pred = y_pred[:-1]
y_test = y_test[:-1]
p = np.argmax(y_pred, axis=-1)
y_orig = flatten_predictions(y_test)
y_preds = flatten_predictions(p)
print(classification_report(y_orig, y_preds))
print(len(y_orig))
print(len(y_preds))

predictions = from_num_to_class(p, idx2tag)

all_outputs, all_outputs_per_sentence = assemble_predictions(predictions, X1_test, sentences_test, param_dict["max_len"])
all_outputs_per_sentence_alt = split_keyphrases(all_outputs_per_sentence)

with open("all_outputs_per_sentence_new.json", "w") as outfile:
    json.dump(all_outputs_per_sentence, outfile)

with open("all_outputs_per_sentence_alt_new.json", "w") as outfile:
    json.dump(all_outputs_per_sentence_alt, outfile) 
"""
