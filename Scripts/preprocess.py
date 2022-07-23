import pandas as pd
import string
import re
from nltk.tokenize import regexp_tokenize
from nltk import pos_tag
from nltk import RegexpParser
from nltk.chunk import conlltags2tree, tree2conlltags


def replace(text):
    """
    Accepts a text string and:
     - replaces html indicators for a new line with a 'NEW_LINE' indicator
     - adds empty space after a '-' so they will not be tokenized as part of a word
     - removes url addresses from the text
     - removes "Request" string in the beginning since it's not informative
    """

    text = re.sub("<br/>", " NEW_LINE ", text)
    text = re.sub("<br>", " NEW_LINE ", text)
    text = re.sub("-", "- ", text)
    url_regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|''[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    text = re.sub(url_regex, "", text)
    text = re.sub("[\[\{\(][Rr][Ee][Qq][\w]*[\]\}\)]", "", text)
    text = re.sub("[\[\{\(][Ss][Uu][Gg][Gg][Ee][Ss][Tt][/\w]*[\]\}\)]", "", text)
    text = re.sub("^ ", "", text)
    return text


def tokenizer(submissions):
    """
    Accepts a list of strings and for each one returns its tokens,
    corresponding pos tags and chunk tags. Returns a dataframe
    with the tokens, their tags and the submission indicators.
    """
    sentences_tokenized = []
    splitted_words_all = []
    pos_tags = []
    chunk_list = []
    curr_id = []
    id = 0
    for submission in submissions:

        # forms tokens out of alphabetic sequences, money expressions,
        # and any other non-whitespace sequences
        tokens = regexp_tokenize(submission, pattern='\w+|\$[\d\.]+|\S+')

        # forms tokens with removing the punctuation
        tags = pos_tag(tokens)

        pattern = 'NP: {<DT>?<JJ>*<NN>}'
        chunker = RegexpParser(pattern)
        chunks = chunker.parse(tags)
        tagged_chunks = tree2conlltags(chunks)

        for token in tokens:
            splitted_words_all.append(token)
            curr_id.append(id)
        sentences_tokenized.append(tokens)
        for tag in tags:
            pos_tags.append(tag[1])
        for chunk in tagged_chunks:
            chunk_list.append(chunk[2])
        id += 1

    sentence_indicators = []
    index = 0
    for sentence in sentences_tokenized:
        for word in sentence:
            sentence_indicators.append("Sentence " + str(index))
        index = index + 1

    t = {'Sentence': pd.Series(sentence_indicators),
         'sent_id': pd.Series(curr_id),
         'Words': pd.Series(splitted_words_all),
         'POS_tag': pd.Series(pos_tags),
         'Chunk_tag': pd.Series(chunk_list)
         }

    final_data = pd.DataFrame(t)
    return final_data

