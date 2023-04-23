import pandas as pd
import numpy as np
from tqdm import tqdm, trange
import string
import os
#import matplotlib.pyplot as plt
from keras.preprocessing.sequence import pad_sequences

from torch import cuda
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler

#import transformers
from transformers import AdamW, get_linear_schedule_with_warmup
from transformers import BertTokenizer, BertConfig, BertForTokenClassification
from transformers import RobertaTokenizer, RobertaForTokenClassification
#from transformers import XLMRobertaForTokenClassification, XLMRobertaTokenizer
#from transformers import FunnelTokenizer, FunnelForTokenClassification
#from transformers import DebertaTokenizer, DebertaForTokenClassification

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from seqeval.metrics import f1_score, accuracy_score


def read_data():
    #os.chdir('D:/TU_Graz/Thesis/Datasets/Reddit_features')
    train = pd.read_csv("../project/data/train_final_all.csv")
    test = pd.read_csv("../project/data/test_final_all.csv")
    data = train.append(test)

    return train, test, data


def group_sentences(data, category):
    all_sents = []
    sent_ids = data['Sent_id'].unique()
    for curr_id in sent_ids:
        tmp_df = data[data['Sent_id'] == curr_id]
        tmp_df = pd.concat([tmp_df['Token'], tmp_df["Token_index"], tmp_df.iloc[:,4:147], tmp_df[category]], axis = 1)
        records = tmp_df.to_records(index=False)
        all_sents.append(records)
    return all_sents


def remove_sents_over_threshold(sents, threshold):
    sentences = list()
    for s in sents:
        if len(s) < threshold:
            sentences.append(s)
    return sentences


def set_processor_params():
    device = 'cuda' if cuda.is_available() else 'cpu'
    #n_gpu = cuda.device_count()
    #cuda.get_device_name(0)
    return device


def tokenize(sentence, sentence_labels):
    tokenized_sentence = []
    labels = []
    for word, label in zip(sentence, sentence_labels):
        str_word = str(word)
        tokenized_word = tokenizer.tokenize(str_word) # Tokenize the word
        n_subwords = len(tokenized_word) # Count subwords
        tokenized_sentence.extend(tokenized_word) # Add to the final tokenized list
        labels.extend([label] * n_subwords) # Add the same label of the original word to all of its subwords
    return tokenized_sentence, labels

train, test, data = read_data()
#data_stats(data)
device = set_processor_params()

tag_values = list(set(train["BIO"].values))
tag_values.append("PAD")
tag2idx = {t: i for i, t in enumerate(tag_values)}
idx2tag = {v: k for k, v in tag2idx.items()}

print("prepare sent")
sents = group_sentences(train, 'BIO')
sents = remove_sents_over_threshold(sents, 300)
sentences = [[word[0] for word in sentence] for sentence in sents]
labels = [[tag2idx[w[len(w)-1]] for w in s] for s in sents]

MAX_LEN = 350
BATCH_SIZE = 4
EPOCHS = 15
LEARNING_RATE = 3e-5
#LEARNING_RATE = 0.00003
MAX_GRAD_NORM = 1.0

print("tokenize")
tokenizer = RobertaTokenizer.from_pretrained("roberta-large", do_lower_case=False)

#tokenizer = BertTokenizer.from_pretrained("bert-base-multilingual-cased", do_lower_case=False)
tokenized_texts_and_labels = [tokenize(sentence, sentence_labels) for sentence, sentence_labels in zip(sentences, labels)]

tokenized_texts = [token_label_pair[0] for token_label_pair in tokenized_texts_and_labels]
labels_subwords = [token_label_pair[1] for token_label_pair in tokenized_texts_and_labels]

print("pad")
# Cut the token and label sequences to the max length
input_ids = pad_sequences([tokenizer.convert_tokens_to_ids(txt) for txt in tokenized_texts], maxlen = MAX_LEN, dtype="long", value=0.0,
                          truncating="post", padding="post")
input_tags = pad_sequences([[l for l in lab] for lab in labels_subwords], maxlen = MAX_LEN, value = tag2idx["PAD"],
                           padding="post", dtype="long", truncating="post")
attention_masks = [[float(i != 0.0) for i in ii] for ii in input_ids]

tr_inputs, val_inputs, tr_tags, val_tags = train_test_split(input_ids, input_tags, test_size=0.1, random_state=2021)
tr_masks, val_masks, _, _ = train_test_split(attention_masks, input_ids, test_size=0.1, random_state=2021)

from torch import tensor
tr_inputs = tensor(tr_inputs)
val_inputs = tensor(val_inputs)
tr_tags = tensor(tr_tags)
val_tags = tensor(val_tags)
tr_masks = tensor(tr_masks)
val_masks = tensor(val_masks)

train_data = TensorDataset(tr_inputs, tr_masks, tr_tags)
train_sampler = RandomSampler(train_data)
train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=BATCH_SIZE)
valid_data = TensorDataset(val_inputs, val_masks, val_tags)
valid_sampler = SequentialSampler(valid_data)
valid_dataloader = DataLoader(valid_data, sampler=valid_sampler, batch_size=BATCH_SIZE)

# Pretrained model params
model = RobertaForTokenClassification.from_pretrained("roberta-large", num_labels=len(tag2idx), output_attentions=False, output_hidden_states=False)

#model = BertForTokenClassification.from_pretrained("bert-base-multilingual-cased", num_labels = len(tag2idx), output_attentions = False, output_hidden_states=False)
#model.cuda() # Pass the model parameters to gpu

FULL_FINETUNING = False
if FULL_FINETUNING:
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'gamma', 'beta']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
         'weight_decay_rate': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
         'weight_decay_rate': 0.0}
    ]
else:
    param_optimizer = list(model.classifier.named_parameters())
    optimizer_grouped_parameters = [{"params": [p for n, p in param_optimizer]}]

optimizer = AdamW(
    optimizer_grouped_parameters,
    lr=LEARNING_RATE,
    eps=1e-8
)

# Total number of training steps is number of batches * number of epochs.
total_steps = len(train_dataloader) * EPOCHS

# Create the learning rate scheduler.
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)

from torch import no_grad

loss_values, validation_loss_values = [], []
accuracy_values, validation_accuracy_values = [], []

for i in trange(EPOCHS, desc="Epoch"):

    # TRAINING
    # Perform one full pass over the training set
    model.train()  # Put the model into training mode
    total_loss, total_accuracy = 0, 0  # Reset the total loss and acc. for current epoch

    # Training loop
    for step, batch in enumerate(train_dataloader):
        batch = tuple(t.to(device) for t in batch)  # add batch to gpu
        b_input_ids, b_input_mask, b_labels = batch  # Input ids, mask and labels of the current batch
        model.zero_grad()  # Always clear any previously calculated gradients before performing a backward pass
        # cuda.empty_cache()
        # Forward pass
        # This will return the loss (rather than the model output) because we have provided the `labels`.
        outputs = model(b_input_ids, attention_mask=b_input_mask, labels=b_labels)
        loss = outputs[0]

        # Perform a backward pass to calculate the gradients
        loss.backward()
        total_loss += loss.item()  # track train loss

        # Clip the norm of the gradient to help prevent the exploding gradients problem
        from torch.nn.utils import clip_grad_norm_

        clip_grad_norm_(parameters=model.parameters(), max_norm=MAX_GRAD_NORM)

        optimizer.step()  # update parameters
        scheduler.step()  # Update the learning rate

    avg_train_loss = total_loss / len(train_dataloader)  # Calc. avg loss over training data
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
        # cuda.empty_cache()
        # Telling the model not to compute or store gradients, to save memory and speed up validation
        with no_grad():
            cuda.empty_cache()
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

# plot_learning_curves()

print("Prepare test set...")
test_sents = group_sentences(test, 'BIO')
test_sents = remove_sents_over_threshold(test_sents, 300)
test_sentences = [[word[0] for word in sentence] for sentence in test_sents]
test_labels = [[tag2idx[w[len(w)-1]] for w in s] for s in test_sents]
#test_sentences = [" ".join(sentence) for sentence in test_sentences]
#true_labels = [[tag for w in s] for s in test_labels]
test_labels_str = [[w[len(w)-1] for w in s] for s in test_sents]

print("Tokenize and predict...")
all_predictions = []
all_true_labels = []

all_predictions_list = []

for lab in test_labels:
    all_true_labels.extend(lab)

for test_sentence in test_sentences:
    tokenized_sentence = tokenizer.encode(test_sentence)
    input_ids = tensor([tokenized_sentence])
    #input_ids = tensor([tokenized_sentence]).cuda()
    #input_masks = tensor([[float(i != 0.0) for i in ii] for ii in input_ids]).cuda()
    input_masks = tensor([[float(i != 0.0) for i in ii] for ii in input_ids])
    with no_grad():
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
report = classification_report(all_true_labels, all_preds)
print(report)