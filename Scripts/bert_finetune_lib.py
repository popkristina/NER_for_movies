"""
Script not completed or tested
"""

import transformers
from transformers import AdamW, get_linear_schedule_with_warmup
from transformers import BertTokenizer, BertConfig, BertForTokenClassification
from transformers import RobertaTokenizer, RobertaForTokenClassification
from transformers import XLMRobertaForTokenClassification, XLMRobertaTokenizer
from transformers import FunnelTokenizer, FunnelForTokenClassification
from transformers import DebertaTokenizer, DebertaForTokenClassification

from keras.preprocessing.sequence import pad_sequences
import torch
from torch import cuda
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from sklearn.model_selection import train_test_split


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

    return input_ids, input_tags, attention_masks


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
    elif model_name == "mobile":
        model = \
            MobileBertForTokenClassification.from_pretrained('google/mobilebert-cased', num_labels=len(tag2idx),
                                                             output_attentions=False, output_hidden_states=False)
    elif model_name == "bert_mlt_unc":
        model = \
            BertForTokenClassification.from_pretrained("bert-base-multilingual-uncased", num_labels=len(tag2idx),
                                                       output_attentions=False, output_hidden_states=False)
    model.cuda()
    return model


def split_and_convert_to_tensor(input_ids, input_tags, attention_masks):
    """
    Accepts a list of inputs in form ids, their associated tags,
    and attention masks. Splits them with train-test-split and
    converts them to torch tensors.
    """
    train_inputs, valid_inputs, train_tags, valid_tags = \
        train_test_split(input_ids, input_tags, test_size=0.2)
    train_masks, valid_masks, _, _ = \
        train_test_split(attention_masks, input_ids, test_size=0.2)

    train_inputs = torch.tensor(train_inputs)
    train_tags = torch.tensor(train_tags)
    train_masks = torch.tensor(train_masks)

    valid_inputs = torch.tensor(valid_inputs)
    valid_tags = torch.tensor(valid_tags)
    valid_masks = torch.tensor(valid_masks)

    return train_inputs, train_tags, train_masks, valid_inputs, valid_tags, valid_masks


def prepare_bert_data(input_ids, input_tags, attention_masks, batch_size):
    """
    Accepts a list of input ids, tags and attention masks.
    First splits them to train and validation data and turns
    them into tensors, then creates a DataLoader() object
    out of them.
    """
    tr_inputs, tr_tags, tr_masks, val_inputs, val_tags, val_masks =\
        split_and_convert_to_tensor(input_ids, input_tags, attention_masks)
    train_data = TensorDataset(tr_inputs, tr_masks, tr_tags)
    train_sampler = RandomSampler(train_data)
    train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size)

    valid_data = TensorDataset(val_inputs, val_masks, val_tags)
    valid_sampler = SequentialSampler(valid_data)
    valid_dataloader = DataLoader(valid_data, sampler=valid_sampler, batch_size=batch_size)

    return train_dataloader, valid_dataloader


def set_optimizer_params(train_dataloader, model, learning_rate, epochs):
    """
    Sets the optimizer parameters
    """
    # Set optimizer parameters
    param_optimizer = list(model.named_parameters())
    # optimizer_grouped_parameters = [{"params": [p for n, p in param_optimizer]}]
    no_decay = ['bias', 'gamma', 'beta']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
         'weight_decay_rate': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
         'weight_decay_rate': 0.0}]
    optimizer = AdamW(optimizer_grouped_parameters, lr=learning_rate, eps=1e-8)

    # Total number of training steps is number of batches * number of epochs.
    total_steps = len(train_dataloader) * epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0,
                                                num_training_steps=total_steps)

    return optimizer, scheduler


def set_processor_params():
    """
    Needed for bert
    """
    device = 'cuda' if cuda.is_available() else 'cpu'
    n_gpu = torch.cuda.device_count()
    torch.cuda.get_device_name(0)
    return device, n_gpu


def plot_learning_bert_curves():
    sns.set(style='darkgrid')
    sns.set(font_scale=1.5)
    plt.rcParams["figure.figsize"] = (6, 6)

    # plt.plot(loss_values, 'b-o', label="training loss")
    # plt.plot(validation_loss_values, 'r-o', label="validation loss")
    plt.plot(validation_accuracy_values, 'g-o', label="validation accuracy")
    plt.title("Learning curve")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.show()


def train_bert(sentences, labels, epochs, model_name, tag2idx, max_len, learning_rate, max_grad_norm):

    """
    Fine-tune the model
    """
    device, n_gpu = set_processor_params()

    # Initialize new empty lists to keep track of accuracy and loss
    loss_values, validation_loss_values = [], []
    accuracy_values, validation_accuracy_values = [], []

    # Preprocess data format, get model and set parameters
    input_ids, input_tags, attention_masks = padding(sentences, labels, model_name, max_len)
    train_dataloader = prepare_bert_data(input_ids, input_tags, attention_masks, batch_size)
    model = select_model(model_name, tag2idx)
    optimizer, scheduler = set_optimizer_params(train_dataloader, model, learning_rate, epochs)

    for i in trange(epochs, desc="Epoch"):

        # TRAINING
        # Perform one full pass over the training set
        # Put model into training mode and reset the total loss and acc. for current epoch
        model.train()
        total_loss, total_accuracy = 0, 0

        # Training loop
        for step, batch in enumerate(train_dataloader):
            # Add batch to gpu; Get input ids, mask and labels of the current batch
            batch = tuple(t.to(device) for t in batch)
            b_input_ids, b_input_mask, b_labels = batch
            model.zero_grad()

            # Forward pass
            # Return the loss (rather than the model output)
            outputs = model(b_input_ids, attention_mask=b_input_mask, labels=b_labels)
            loss = outputs[0]

            # Perform backward pass to calculate gradients and track tr. loss
            loss.backward()
            total_loss += loss.item()

            # Clip the norm of the gradient to prevent the exploding gradients problem
            torch.nn.utils.clip_grad_norm_(parameters=model.parameters(), max_norm=max_grad_norm)

            # Update parameters and learning rate
            optimizer.step()
            scheduler.step()

        # Calc. avg loss over training data
        avg_train_loss = total_loss / len(train_dataloader)
        print("Average train loss: {}".format(avg_train_loss))
        loss_values.append(avg_train_loss)  # Store the loss value for plotting the learning curve

        # VALIDATION
        # After the completion of each training epoch, measure performance on validation set
        model.eval()  # Put the model into evaluation mode
        eval_loss, eval_accuracy = 0, 0  # Reset the validation loss for current epoch
        nb_eval_steps, nb_eval_examples = 0, 0
        predictions, true_labels = [], []

        # Validation loop
        for batch in valid_dataloader:
            batch = tuple(t.to(device) for t in batch)
            b_input_ids, b_input_mask, b_labels = batch

            # Telling the model not to compute or store gradients, to save memory and speed up validation
            with torch.no_grad():
                # Forward pass, calculate logit predictions
                # This will return the logits rather than the loss because we have not provided labels
                outputs = model(b_input_ids, attention_mask=b_input_mask, labels=b_labels)

                # encoded_input = tokenizer(text, return_tensors='pt')
                # output = model(**encoded_input)

            logits = outputs[1].detach().cpu().numpy()  # Move logits to cpu
            label_ids = b_labels.to('cpu').numpy()  # Move labels to cpu
            eval_loss += outputs[0].mean().item()  # Valid. loss for current batch

            predictions.extend([list(p) for p in np.argmax(logits, axis=2)])
            true_labels.extend(label_ids)

        eval_loss = eval_loss / len(valid_dataloader)
        validation_loss_values.append(eval_loss)
        print("Validation loss: {}".format(eval_loss))

        # Calculate the accuracy for this batch of test sentences
        pred_tags = [tag_values[p_i] for p, l in zip(predictions, true_labels)
                     for p_i, l_i in zip(p, l) if tag_values[l_i] != "PAD"]
        valid_tags = [tag_values[l_i] for l in true_labels
                      for l_i in l if tag_values[l_i] != "PAD"]
        validation_accuracy_values.append(accuracy_score(pred_tags, valid_tags))
        print("Validation Accuracy: {}".format(accuracy_score(pred_tags, valid_tags)))

    return model


def bert_test(model, test_sentences, test_labels, model_name):

    """
    Test the fine-tuned model.
    """
    all_predictions = []
    all_true_labels = []

    tokenizer = select_tokenizer(model_name)
    all_predictions_list = []

    for lab in test_labels:
        all_true_labels.extend(lab)

    for test_sentence in test_sentences:
        tokenized_sentence = tokenizer.encode(test_sentence)
        input_ids = torch.tensor([tokenized_sentence]).cuda()
        input_masks = torch.tensor([[float(i != 0.0) for i in ii] for ii in input_ids]).cuda()
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
        all_predictions.extend(new_labels)
        all_predictions_list.append(new_labels)
    all_preds = [tag2idx[label] for label in all_predictions]
    return all_predictions, all_predictions_list, all_preds

