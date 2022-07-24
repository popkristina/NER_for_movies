import pandas as pd
import spacy
import pytextrank
import nltk
import json
from rake_nltk import Rake

nlp = spacy.load("en_core_web_sm")
nlp.add_pipe("textrank")

algorithm = 'textrank' # rake or textrank
path = '../data/'
submissions = pd.read_csv(path + 'test_submissions_simplified.csv')


def rake(sentences):
    keyphrases = dict()
    r = Rake()
    for index, row in sentences.iterrows():
        r.extract_keywords_from_text(row['text'])
        if row['id'] not in keyphrases.keys():
            keyphrases[row['id']] = list()
        keyphrases[row['id']].append(r.get_ranked_phrases())
    return keyphrases


def textrank(sentences):
    keyphrases = dict()
    for index, row in sentences.iterrows():
        doc = en_nlp(row['text'])
        if row['id'] not in keyphrases.keys():
            keyphrases = list()
        phrases = [phrase.text for phrase in doc._.phrases]
        print(phrases)
        keyphrases.extend(phrases)
    return keyphrases


if algorithm == 'rake':
    keyphrases = rake(submissions)
    with open("../predictions/" + "rake_keywords_format_1.json", "w") as outfile:
        json.dump(keyphrases, outfile)
else:
    keyphrases = textrank(submissions)

print(len(keyphrases))