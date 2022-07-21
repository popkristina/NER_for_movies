
"""
Option 1: Input one submission
"""
input_submission = 'Please recommend me some movies like: The Fault in Our Stars. I ' \
                   'like actors like Logan Lerman and Mischa Burton. Please no horrors ' \
                   'or thrillers.'

"""
Option 2: Input datasets
"""

# TODO: Add Input dataset

# 1. PREPROCESS AND TOKENIZE INPUT TEXT
text = replace(input_submission)
texts = [text]
tokenized_text = tokenizer(texts)  # Tokenized text is in dataframe format

# 2. GROUP SENTENCES TO GET A LIST OF TOKENS, LABELS AND SENTENCE IDS
sentences, labels, sent_ids = group_sents(tokenized_text)

# 3. EXTRACT FEATURES FROM TEXT
if features:
    feats = extract_all_feats(sentences, sent_ids)

else:
    # load json and create model
    #json_file = open("models/" + model_name + ".json", "r")
    #loaded_model_json = json_file.read()
    #json_file.close()
    #model = tf.keras.models.model_from_json(loaded_model_json)

    param_dict["batch_size"] = 2
    if features:
        model = build_model(param_dict["max_len"], n_tags)
    else:
        model = build_elmo_model(param_dict["max_len"], n_tags)

    model.compile(optimizer=param_dict["optimizer"], loss=param_dict["loss"], metrics=param_dict["metrics"])
    model.load_weights("models/" + model_name + ".h5")


"""
############ 7 TEST THE MODEL WITH THE TEST SET ################## 
"""
sentences_test = group_sentences(test_set, "Sent_id", "BIO")
for i in range(0, len(sentences_test)):
    sentences_test[i] = sentences_test[i][0:300]
#sentences_test = [s for s in sents_test if len(s) <= param_dict["max_len"]]
sentences_test.append(sentences_test[-1])
print(len(sentences_test))

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