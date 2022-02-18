import transformers
from transformers import AdamW, get_linear_schedule_with_warmup
from transformers import BertTokenizer, BertConfig, BertForTokenClassification
from transformers import RobertaTokenizer, RobertaForTokenClassification
from transformers import XLMRobertaForTokenClassification, XLMRobertaTokenizer
from transformers import FunnelTokenizer, FunnelForTokenClassification
from transformers import DebertaTokenizer, DebertaForTokenClassification


def select_tokenizer(str):
    """
    Select a pre-trained BERT tokenizer based
    on previously selected model
    """
    if str == "funnel":
        tokenizer = \
            FunnelTokenizer.from_pretrained('funnel-transformer/medium')
    elif str == "roberta_base":
        tokenizer = \
            RobertaTokenizer.from_pretrained('roberta-base')
    elif str == "bert_multi_unc":
        tokenizer = \
            BertTokenizer.from_pretrained("bert-base-multilingual-uncased", do_lower_case=True)
    elif str == "bert_multi_cs":
        tokenizer = \
            BertTokenizer.from_pretrained("bert-base-multilingual-cased", do_lower_case=False)
    elif str == "distil-multi":
        tokenizer = \
            DistilBertTokenizer.from_pretrained('distilbert-base-multilingual-uncased', do_lower_case=True)
    elif str == "deberta":
        tokenizer = \
            DebertaTokenizer.from_pretrained('microsoft/deberta-base')
    return tokenizer


def tokenize(sentence, sentence_labels, str_model):
    """
    Accepts a list of tokenized sentences and their
    corresponding token labels, and a string for the
    pre-trained model. Based on the 'str_model', it
    retrieves a pre-trained word-piece tokenizer.

    Counts the number of subwords that the words
    get split into, and extends the corresponding
    labels of the original word to the word pieces.
    """
    tokenized_sentence = []
    labels = []
    tokenizer = select_tokenizer(str_model)
    for word, label in zip(sentence, sentence_labels):
        str_word = str(word)
        tokenized_word = tokenizer.tokenize(str_word)
        n_subwords = len(tokenized_word)
        tokenized_sentence.extend(tokenized_word)
        labels.extend([label] * n_subwords)
    return tokenized_sentence, labels

