"""
Option 1: Input one submission
"""
input_submission = 'Please recommend me some movies like: The Fault in Our Stars. I ' \
                   'like actors like Logan Lerman and Mischa Burton. Please no horrors ' \
                   'or thrillers.'

"""
Option 2: Input dataset
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