from keras.preprocessing.sequence import pad_sequences
import pandas as pd
import numpy as np
from collections import Counter
import string
from difflib import SequenceMatcher
from sklearn.model_selection import train_test_split
import json


def pad_textual_data(sentences, max_len):
    """
    Accepts a list of tokenized sentences and
    pads each sentence to 'max_len' length.
    """
    x = [[w[1] for w in s] for s in sentences]

    new_x = []
    for seq in x:
        new_seq = []
        for i in range(max_len):
            try:
                new_seq.append(seq[i])
            except:
                new_seq.append("__PAD__")
        new_x.append(new_seq)
    x = new_x
    return x


def pad_textual_char_data(sentences, max_len, max_len_char, n_chars, char2idx):
    """
    Accepts a list of tokenized sentences and
    pads each sentence to 'max_len' length and
    pads each word in a sentence to 'max_len_char'.
    """
    x = []
    for sentence in sentences:
        padded_sentence = []
        for i in range(max_len):
            padded_word = []
            for j in range(max_len_char):
                try:
                    padded_word.append(char2idx.get(sentence[i][0][j]))
                except:
                    padded_word.append(n_chars)
            padded_sentence.append(padded_word)
        x.append(np.array(padded_sentence))
    return x


def pad_feature_data(sentences, max_len, num_feats):
    """
    Accepts a list of tokenized sentences and their features.
    Pads each sentence to 'max_len' length and the vector of
    its features to 'max_len' vectors of zeros.
    """
    x = []
    for sentence in sentences:
        sent_ft = list()
        for word in sentence:
            ft = list()
            for i in range(2, num_feats+2):
                ft.append(word[i])
            sent_ft.append(ft)
        for j in range(len(sentence) - 1, max_len - 1):
            ft = list()
            for i in range(1, num_feats+1):
                ft.append(0)
            sent_ft.append(ft)
        x.append(sent_ft)
    return x


def create_alt_movie_dict(movies_matched):
    """
    Accepts a dataframe in the form:

    ####################################################
    ##### id ##### original_name ##### alternative #####
    ####################################################

    and returns a dictionary where each key is an original
    title name and its corresponding values are the
    alternative names (in alternative languages)

    """
    alt_names = dict()
    for index, row in movies_matched.iterrows():
        if row["original_title"] not in alt_names.keys():
            alt_names[row["original_title"]] = list()
        alt_names[row["original_title"]].append(row["alternative"])
    return alt_names


def read_preprocessed_data(path):
    """
    Read preprocessed train and test data in "all features"
    format". Every row represents one token, its sentence
    identification, its token identification and its other
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


def create_dict(str_list, reverse=False):
    """
    Accepts a list of strings and returns a dictionary
    where each string has an associated number.
    If reverse is sat to true, the numbers become keys
    and the strings become values.
    """
    if reverse:
        return {i: st for i, st in enumerate(str_list)}
    return {st: i for i, st in enumerate(str_list)}


def group_sentences(data, sent_identificator, category):
    """
    Accepts the tokenized data with features,
    and creates a list where every element is another
    list for the current tokenized sentence, and it
    consists of a list of features for every token that
    comprise the sentence.
    """
    all_sents = []
    sent_ids = data[sent_identificator].unique()
    for curr_id in sent_ids:
        tmp_df = data[data[sent_identificator] == curr_id]
        tmp_df = pd.concat(
            [tmp_df['Sent_id'], tmp_df['Token'], tmp_df["Token_index"], tmp_df.iloc[:, 4:44], tmp_df.iloc[:, 137:147],
             tmp_df[category]], axis=1)
        records = tmp_df.to_records(index=False)
        all_sents.append(records)

    return all_sents


def group_sents(submissions_tokenized):
    """
    Groups tokenized sentences in a list-of-lists form
    with a Sentence Getter object.
    """
    getter = SentenceGetter(submissions_tokenized)
    sentences = [[word[0] for word in sentence] for sentence in getter.sentences]
    labels = [[s[1] for s in sentence] for sentence in getter.sentences]
    sent_ids = [[s_id[3] for s_id in sentence] for sentence in getter.sentences]
    return sentences, labels, sent_ids


class SentenceGetter(object):
    """
    Creates an object out of a dataframe with tokenized
    data and its features. Groups them into list of lists
    format where they are grouped by their sentence id.
    """
    def __init__(self, data):
        self.n_sent = 1
        self.data = data
        self.empty = False
        agg_func = lambda s: [(w, p, c, s) for w, p, c, s in
                    zip(s["Words"].values.tolist(),
                        s["POS_tag"].values.tolist(),
                        s["Chunk_tag"].values.tolist(),
                        s["sent_id"].values.tolist())]
        self.grouped = self.data.groupby("Sentence").apply(agg_func)
        self.sentences = [s for s in self.grouped]

    def get_next(self):
        """
        In a grouped object of sentences,
        returns the next sentence
        """
        try:
            s = self.grouped
            self.n_sent += 1
            return s
        except:
            return None


def group(data, category):
    all_sents = []
    sent_ids = data['Sent_id'].unique()
    for curr_id in sent_ids:
        tmp_df = data[data['Sent_id'] == curr_id]
        tmp_df = pd.concat([tmp_df['Token'], tmp_df["Token_index"], tmp_df.iloc[:, 4:44],
                            tmp_df.iloc[:, 137:147], tmp_df[category]], axis=1)
        records = tmp_df.to_records(index=False)
        all_sents.append(records)
    return all_sents


def remove_sents_over_threshold(sents, threshold):
    """
    Given a list of sentences and a
    threshold, return only the ones
    that have more words than the
    threshold.
    """
    sentences = list()
    for s in sents:
        if len(s) < threshold:
            sentences.append(s)
    return sentences


def data_stats(data):
    """
    Accepts a dataframe and returns counts of its categories
    """
    frequencies = data.BIO.value_counts()
    tags = {}
    for tag, count in zip(frequencies.index, frequencies):
        if tag != "O":
            # if tag[2:] not in tags.keys():
            tags[tag[2:]] = count
        else:
            tags[tag[2:]] += count
        continue

    print("Number of tags: {}".format(len(data.BIO.unique())))
    print("Tag frequencies: {}".format(frequencies))
    print("Categories: ")
    print(sorted(tags.items(), key=lambda x: x[1], reverse=True))


def split_to_fit_batch(x1, y, bs, x2="", reshape_y=True):
    """
    Split train dataset and its labels
    to train and validation dataset such
    that it is divisible with the defined
    batch size (required for lstm problems)
    """

    x1_train, x1_valid, y_train, y_valid = train_test_split(x1, y, test_size=0.2, random_state=2021)
    x1_train = x1_train[:(len(x1_train) // bs) * bs]
    x1_valid = x1_valid[:(len(x1_valid) // bs) * bs]

    y_train = y_train[:(len(y_train) // bs) * bs]
    y_valid = y_valid[:(len(y_valid) // bs) * bs]

    if reshape_y:
        y_train = y_train.reshape(y_train.shape[0], y_train.shape[1], 1)
        y_valid = y_valid.reshape(y_valid.shape[0], y_valid.shape[1], 1)

    if x2 is not "":
        x2_train, x2_valid, _, _ = train_test_split(x2, y, test_size=0.2, random_state=2021)
        x2_train = x2_train[:(len(x2_train) // bs) * bs]
        x2_valid = x2_valid[:(len(x2_valid) // bs) * bs]
        return x1_train, x1_valid, x2_train, x2_valid, y_train, y_valid

    return x1_train, x1_valid, y_train, y_valid


def save_as_json(dict, name):
    """
    Accepts a word-to-index map in a dict format and
    a name for the map. Save it to json file.
    """
    dict_save = open("./helper_dicts/" + name + ".json", "w")
    json.dump(dict, dict_save)
    dict_save.close()
    # word2idx_save = open("helper_dicts/w2idx.json", "w")  # save it for further use
    # json.dump(word2idx, word2idx_save)
    # word2idx_save.close()

    # tag2idx_save = open("helper_dicts/t2idx.json", "w")  # save it for further use
    # json.dump(tag2idx, tag2idx_save)
    # tag2idx_save.close()


def create_word_and_tag_list(data):
    """
    Accepts a dataframe with tokenized sentences,
    and returns a list of all od the distinct words
    and tags in the dataset.
    """
    words = list(set(data["Token"].values))
    words.append('ENDPAD')
    n_words = len(words)
    tags = list(set(data['BIO'].values))
    n_tags = len(tags)
    return words, n_words, tags, n_tags


def from_num_to_class(p, idx2tag):
    """
    Turn the predictions from numerical
    to their class names.
    """
    predictions = []
    for sent in p:
        sent_categories = []
        for num in sent:
            sent_categories.append(idx2tag[str(num)])
        predictions.append(sent_categories)
    return predictions


def read_sentences_from_json(json_file):
    print()
    # implement

def read_sentences_from_txt(txt_file):
    print()

def read_sentences_from_obj(obj_file):
    print()