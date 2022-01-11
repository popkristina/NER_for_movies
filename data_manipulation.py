from keras.preprocessing.sequence import pad_sequences
import pandas as pd
import numpy as np
from collections import Counter
import string
from difflib import SequenceMatcher


def group_sentences(data, category=""):
    all_sents = []
    sent_ids = data['Sent_id'].unique()
    for curr_id in sent_ids:
        tmp_df = data[data['Sent_id'] == curr_id]
        if len(category) > 1:
            tmp_df = pd.concat([tmp_df['Token'], tmp_df["Token_index"], tmp_df.iloc[:, 4:44], tmp_df[category]], axis=1)
        else:
            tmp_df = pd.concat([tmp_df['Token'], tmp_df["Token_index"], tmp_df.iloc[:, 4:44]], axis=1)
        records = tmp_df.to_records(index=False)
        all_sents.append(records)
    return all_sents


def group_sents(submissions_tokenized):
    getter = SentenceGetter(submissions_tokenized)
    sentences = [[word[0] for word in sentence] for sentence in getter.sentences]
    labels = [[s[1] for s in sentence] for sentence in getter.sentences]
    sent_ids = [[s_id[3] for s_id in sentence] for sentence in getter.sentences]
    return sentences, labels, sent_ids


class SentenceGetter(object):

    def __init__(self, data):
        self.n_sent = 1
        self.data = data
        self.empty = False
        agg_func = lambda s: [(w, p, c, s) for w, p, c, s in zip(s["Words"].values.tolist(),
                                                                 s["POS_tag"].values.tolist(),
                                                                 s["Chunk_tag"].values.tolist(),
                                                                 s["sent_id"].values.tolist())]
        self.grouped = self.data.groupby("Sentence").apply(agg_func)
        self.sentences = [s for s in self.grouped]

    def get_next(self):
        try:
            s = self.grouped
            self.n_sent += 1
            return s
        except:
            return None


def pad_textual_data(sentences, max_len):
    x = list()
    for sentence in sentences:
        padded_sentence = []
        for word in sentence:
            padded_sentence.append(word[0])
        for i in range(len(sentence) - 1, max_len - 1):
            padded_sentence.append("__PAD__")
        x.append(padded_sentence)
    return x


def pad_feature_data(sentences, max_len, num_feats):
    x = []
    for sentence in sentences:
        sent_ft = list()
        for word in sentence:
            ft = [word[i] for i in range(1, num_feats-1)]
            sent_ft.append(ft)
        for j in range(len(sentence) - 1, max_len - 1):
            ft = [0] * num_feats
            sent_ft.append(ft)
        x.append(sent_ft)
    return x


def prepare_and_pad(sentences, max_len):
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
            ft = [word[i] for i in range(1, 41)]
            sent_ft.append(ft)
        for j in range(len(sentence) - 1, max_len - 1):
            ft = [0] * 40
            sent_ft.append(ft)
        X2.append(sent_ft)

    return X1, X2


def create_alt_movie_dict(movies_matched):
    """
    Accepts a dataframe in the form:

    ####################################################
    ##### id ##### original_name ##### alternative #####
    ####################################################

    and returns a dictionary where each key is an original title name and its corresponding values are
    the alternative names (in alternative languages)
    """
    alt_names = dict()
    for index, row in movies_matched.iterrows():
        if row["original_title"] not in alt_names.keys():
            alt_names[row["original_title"]] = list()
        alt_names[row["original_title"]].append(row["alternative"])
    return alt_names


def read_preprocessed_data(path):
    """
    Read preprocessed train and test data in "all features" format".
    Every row represents one token, its sentence identification, its token identification and its other
    features.
    """

    train = pd.read_csv(path + "/train_final_all.csv")
    test = pd.read_csv(path + "/test_final_all.csv")
    data = train.append(test)
    return data, train, test


def similarity(string1, string2):
    """
    Returns similarity ratio between two strings.
    Used in case movie written names differ by a little.
    """
    return SequenceMatcher(None, string1, string2).ratio()


def split_based_on_id(data):
    """
    Splits a dataset of submissions to train and test sets
    based on a previously defined set of submission ids that
    should be kept for testing.
    Returns a train and test Pandas dataframes.
    """
    train = pd.DataFrame()
    test = pd.DataFrame()
    for index, row in data.iterrows():
        if row['id'] in test_set_ids_list:
            test = test.append([row])
        else:
            train = train.append([row])
    train.columns = data.columns
    test.columns = data.columns

    return train, test


def fix_vertical_line(df, column):
    """
    Accepts a dataframe of submissions and a column of
    extracted entities separated with '|'.
    Checks if every entry has a '|' in the beginning or
    the end due to previous merge of columns.
    """
    col = []
    for line in df[column]:
        if len(line) == 1:
            line = ""
        if line.startswith('|'):
            line = line[1:]
        if line.endswith('|'):
            line = line[:-1]
        col.append(line)
    return col
