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
    Accepts a tokenized sentence and its token
    labels, and a string for the pre-trained model.
    Based on the 'str_model', it retrieves a
    pre-trained word-piece tokenizer.

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


def tokenize_sentences(sentences, labels, model_name):
    """
    Accepts a list of tokenized sentences and their
    associated labels and the model name. Calls
    function tokenize to piece-wise tokenize each
    sentence separately.
    """

    tokenized_texts_and_labels = [tokenize(sentence, sentence_labels, model_name)
                                  for sentence, sentence_labels in zip(sentences, labels)]
    tokenized_texts = [token_label_pair[0] for token_label_pair in tokenized_texts_and_labels]
    labels_subwords = [token_label_pair[1] for token_label_pair in tokenized_texts_and_labels]

    return tokenized_texts_and_labels, tokenized_texts, labels_subwords


def padding(sentences, labels, model_name, max_len):
    """
    Accepts a list of tokenized sentences and pads
    them to "max_len". Creates input ids, tags
    and attention masks, all padded to "max_len"
    """
    tokenizer = select_tokenizer(model_name)
    _, tok_texts, lab_subwords = tokenize_sentences(sentences, labels, model_name)
    input_ids = pad_sequences([tokenizer.convert_tokens_to_ids(txt) for txt in tok_texts],
                              maxlen=max_len, dtype="long", value=0.0,
                              truncating="post", padding="post")
    input_tags = pad_sequences([[l for l in lab] for lab in lab_subwords],
                               maxlen=max_len, value=tag2idx["PAD"],
                               padding="post", dtype="long", truncating="post")
    attention_masks = [[float(i != 0.0) for i in ii] for ii in input_ids]

    return input_ids, input_tags, input_masks


def select_model(model_name, tag2idx):
    """
    Retrieves a pre-trained model based on
    the model list required with "model name"
    """
    if model_name == "roberta_base":
        model = \
            RobertaForTokenClassification.from_pretrained('roberta-base', num_labels=len(tag2idx),
                                                          output_attentions=False, output_hidden_states=False)
    elif model_name == "funnel":
        model = \
            FunnelForTokenClassification.from_pretrained('funnel-transformer/medium', num_labels=len(tag2idx),
                                                         output_attentions=False, output_hidden_states=False)
    elif model_name == "deberta":
        model = \
            DebertaForTokenClassification.from_pretrained('microsoft/deberta-base', num_labels=len(tag2idx),
                                                          output_attentions=False, output_hidden_states=False)

    return model

def set_optimizer_params():
    return
def set_processor_params():
    """
    Needed for bert
    """
    device = 'cuda' if cuda.is_available() else 'cpu'
    n_gpu = torch.cuda.device_count()
    torch.cuda.get_device_name(0)
    return device, n_gpu
