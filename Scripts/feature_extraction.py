import pandas as pd
from nltk.corpus import stopwords
from nltk.sentiment import SentimentIntensityAnalyzer
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
import en_core_web_sm
from nltk import FreqDist
nlp = en_core_web_sm.load()


def spacy_feats_sentence(index, sentence, sent_id):
    """
    Extracts various SpaCy token-wise features for
    one sentence.
    """
    data = pd.DataFrame()
    string = "Sentence " + str(index)
    whole_sent = " ".join(sentence)
    doc = nlp(whole_sent)

    for X in doc:
        data = data.append(
            [[sent_id[0], string, X.i, X.text, X.tag, X.pos, X.ent_type, X.ent_iob,
              X.lemma, X.norm, X.shape, X.lex_id, X.is_digit, X.is_ascii, X.is_alpha,
              X.is_punct, X.is_left_punct, X.is_right_punct, X.rank, X.is_bracket,
              X.is_space, X.is_quote, X.is_currency, X.is_stop, X.dep, X.lang, X.prob,
              X.sentiment, X.is_lower, X.is_upper, X.like_num, X.is_oov, X.n_lefts,
              X.n_rights, X.is_sent_start, X.has_vector, X.ent_kb_id, X.ent_id, X.lower,
              X.prefix, X.suffix, X.idx, X.cluster, len(X)]])

    data.columns = ['Sent_id', 'Sentence', 'Token_index', 'Token', 'POS_tag', 'POS_universal',
                    'NER_tag', 'NER_iob', 'lemma', 'norm', 'shape', 'lex_id', 'is_digit',
                    'is_ascii', 'is_alpha', 'is_punct', 'is_left_punct', 'is_right_punct',
                    'rank', 'is_bracket', 'is_space', 'is_quote', 'is_currency', 'stopword',
                    'dependency', 'language', 'log_prob', 'sent', 'is_lower', 'is_upper',
                    'like_num', 'out_of_vocab', 'num_lefts', 'num_rights', 'sent_start',
                    'has_vector', 'knowledge_base', 'id_entity', 'lower', 'prefix', 'suffix',
                    'chr_offset', 'brown_cluster', 'num_chars']
    return data


def spacy_feats_all(sentences, sent_ids):
    """
    Extracts various SpaCy token-wise features for
    a list of tokenized sentences.
    """
    data = pd.DataFrame()

    for i in range(0, len(sentences)):
        tmp_df = spacy_feats_sentence(i, sentences[i], sent_ids[i])
        data = data.append([tmp_df])
    data.columns = ['Sent_id', 'Sentence', 'Token_index', 'Token', 'POS_tag', 'POS_universal', 'NER_tag',
                    'NER_iob', 'lemma', 'norm', 'shape', 'lex_id', 'is_digit', 'is_ascii', 'is_alpha',
                    'is_punct', 'is_left_punct', 'is_right_punct', 'rank', 'is_bracket', 'is_space',
                    'is_quote', 'is_currency', 'stopword', 'dependency', 'language', 'log_prob', 'sent',
                    'is_lower', 'is_upper', 'like_num', 'out_of_vocab', 'num_lefts', 'num_rights',
                    'sent_start', 'has_vector', 'knowledge_base', 'id_entity', 'lower', 'prefix',
                    'suffix', 'chr_offset', 'brown_cluster', 'num_chars']

    return data


def spacy_feats_tensors(sentence):
    """
    Extract SpaCy simple word vectors (tensors)
    for one tokenized sentence.
    """
    data = pd.DataFrame()
    whole_sent = " ".join(sentence)
    doc = nlp(whole_sent)
    for X in doc:
        tensors = [item for item in X.tensor]
        data = data.append([tensors])
    return data


def spacy_feats_tensors_all(sentences):
    """
    Extract SpaCy simple word vectors (tensors)
    for a list of tokenized sentences.
    """
    data = pd.DataFrame()

    for i in range(0, len(sentences)):
        tmp_df = spacy_feats_tensors(sentences[i])
        data = data.append([tmp_df])

    # There are 96 tensor features that SpaCy outputs
    tensors = ["Vector_ " + str(i) for i in range(0, 96)]
    data.columns = [tensors]
    return data


def sentiment_feats(sentence, words):
    """
    Extract sentiment features for every token of
    a sentence.
    """
    all_sentence_scores = pd.DataFrame()
    words = words.tolist()
    whole_sent = " ".join(sentence)
    analyzer = SentimentIntensityAnalyzer()
    s_neg_score, s_neu_score, s_pos_score, s_compound_score = \
        analyzer.polarity_scores(whole_sent).values()
    for i in range(0, len(words)):
        w_neg_score, w_neu_score, w_pos_score, w_compound_score = \
            analyzer.polarity_scores(words[i]).values()
        if s_compound_score == 0:
            s_compound_score = 0.0001
        sent_ratio = w_compound_score/s_compound_score
        if i == 0:
            predecessor_sent = 0
        else:
            _, _, _, predecessor_sent = analyzer.polarity_scores(words[i-1]).values()
        if i < len(words)-1:
            _, _, _, successor_sent = analyzer.polarity_scores(words[i+1]).values()
        else:
            successor_sent = 0
        all_sentence_scores = \
            all_sentence_scores.append([[s_compound_score, w_neg_score, w_neu_score, w_pos_score,
                                         w_compound_score, sent_ratio, predecessor_sent, successor_sent]])
    all_sentence_scores.columns = \
        ["Sentence_sent", "Neg_sent_score", "Neu_sent_score", "Pos_sent_score", "Sent_score",
         "word_to_sentence_sent_ratio", "prev_word_sent", "next_word_sent"]
    return all_sentence_scores


def sentiment_feats_all(sentences, df_subset):
    """
    Extract token-wise sentiment features for
    a list of tokenized sentences.
    """
    feats = pd.DataFrame()
    for i in range(0, len(sentences)):
        words_curr_sent = df_subset[df_subset["Sentence"] == "Sentence " + str(i)]["Token"]
        new_feats = sentiment_feats(sentences[i], words_curr_sent)
        feats = feats.append([new_feats])
    feats.columns = \
        ["Sentence sent", "Neg_sent_score", "Neu_sent_score", "Pos_sent_score", "Sent_score",
         "word_to_sentence_sent_ratio", "prev_word_sent", "next_word_sent"]
    return feats


def tf_feats(df_subset):
    """
    Term-frequency features for all tokens in a
    tokenized dataset.
    """
    feats = pd.DataFrame()
    freqdist = FreqDist(df_subset["Token"])
    total_word_count = sum(freqdist.values())
    for word in df_subset["Token"]:
        abs_frequency = freqdist[word]
        normalized_frequency = freqdist[word] / total_word_count
        feats = feats.append([[normalized_frequency, abs_frequency]])
    feats.columns = ["norm_freq", "abs_freq"]
    return feats


def extract_bigrams_trigrams(sentences):
    """
    Bi-gram and trigram extractor from a list
    of tokenized sentences.
    """
    stops = stopwords.words('english')
    stops.append('new_line')
    all_sentences = []
    for sentence in sentences:
        all_sentences.append(" ".join(sentence))
    vectorizer_bigrams = \
        TfidfVectorizer(analyzer="word", ngram_range=(2, 2), tokenizer=None,
                        preprocessor=None, stop_words=stops, max_features=25, max_df=0.9)
    vectorizer_trigrams = \
        TfidfVectorizer(analyzer="word", ngram_range=(3, 3), tokenizer=None,
                        preprocessor=None, stop_words=stops, max_features=25, max_df=0.9)
    bigrams = vectorizer_bigrams.get_feature_names()
    trigrams = vectorizer_trigrams.get_feature_names()
    return bigrams, trigrams


def tf_ngrams(ngrams, words):
    feats = pd.DataFrame()
    n = len(ngrams[0].split(" "))
    words = words.tolist()
    is_predecessor = list()

    if n == 2:
        curr_sent_ngrams = nltk.bigrams(words)
        curr_ngrams = [ngram for ngram in curr_sent_ngrams]
    elif n == 3:
        curr_sent_ngrams = nltk.trigrams(words)
        curr_ngrams = [ngram for ngram in curr_sent_ngrams]

    for ngram in ngrams:
        splits = ngram.split(" ")
        tmp = []
        if n == 2:
            for i in range(0, len(words)):
                counter = 0
                for j in range(0, i):
                    if i >= 2:
                        if j + 2 < i:
                            if (words[j], words[j + 1]) == splits:
                                counter += 1
                tmp.append(counter)
        elif n == 3:
            for i in range(0, len(words)):
                counter = 0
                for j in range(0, i):
                    if i >= 3:
                        if j + 3 < i:
                            if (words[j], words[j + 1], words[j + 2]) == splits:
                                counter += 1
                tmp.append(counter)
        is_predecessor.append(tmp)
    index = 1
    for p in is_predecessor:
        feats[str(index)] = pd.Series(p)
        index = index + 1
    return feats


def tf_ngrams_all(sentences, df_subset, ngrams):
    feats = pd.DataFrame()
    for i in range(0, len(sentences)):
        words_curr_sent = df_subset[df_subset["Sentence"] == "Sentence " + str(i)]["Token"]
        ngram_feats = tf_ngrams(ngrams, words_curr_sent)
        feats = feats.append([ngram_feats])
    names = [str(ngram) + "_is_predecessor" for ngram in ngrams]
    feats.columns = [names]
    return feats


def extract_all_feats(sentences, sent_ids):
    """
    Accepts a set of tokenized sentences
    and returns all of the features that
    can be extracted from every token.
    The corresponding functions for every
    feature type are described within the
    function definition.
    """

    feats1 = spacy_feats_all(sentences, sent_ids)
    feats2 = spacy_feats_tensors_all(sentences)
    feats3 = sentiment_feats_all(sentences, feats1[["Sentence", "Token"]])
    feats4 = tf_feats(feats1[["Sentence", "Token"]])
    feats = pd.concat([feats1, feats2, feats3, feats4], axis=1)
    return feats

