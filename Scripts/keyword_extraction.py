import rake
import pandas as pd
import spacy
import pytextrank

algorithm = 'rake' # rake or textrank
path = './data/'
submissions = pd.read_csv(path + 'submissions.csv')


def rake(sentences):
    keyphrases = []
    r = Rake()
    for text in sentences:
        r.extract_keywords_from_text(text)
        keyphrases.append(r.get_ranked_phrases())
    return keyphrases

def textrank(sentences):
    en_nlp = spacy.load("en_core_web_sm")
    en_nlp.add_pipe("textrank")
    doc = en_nlp(document)

    print()

if algorithm == 'rake':
    rake()
else:
    textrank()
