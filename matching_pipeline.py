import pandas as pd
import json
from difflib import SequenceMatcher

from Scripts.assemble import *
from Scripts.data_manipulation import *

name = 'bert_multilingual_cased_best'

# Load predictions in original format, as per outputted from model
with open("predictions/" + name + "_umatched_format_1.json") as json_file:
    all_predictions = json.load(json_file)

# Fix some predictions
all_predictions = fix_prediction_dicts(all_predictions)

# Change them to format that fits the recommender engine
all_predictions_2 = recommender_format(all_predictions)
with open("predictions/" + name + "_unmatched_format_2.json", "w") as outfile:
    json.dump(all_predictions_2, outfile)

# Read additional data
imdb_genres = pd.read_csv("data/genres.csv", sep=';')  # IMDB genre list
movie_titles = pd.read_csv("data/movie_titles.csv", sep=';', encoding='latin')
movies_matched = pd.read_csv("data/movies_matched.csv")
submissions = pd.read_csv("data/submissions.csv", sep=';')
submissions = submissions.fillna("")

alt_names = create_alt_movie_dict(movies_matched)

# Perform the matching
all_predictions_matched = do_matching(all_predictions, imdb_genres, movie_titles, alt_names)

# Save them as matched dictionary
with open("predictions/" + name + "_matched_format_1.json", "w") as outfile:
    json.dump(all_predictions_matched, outfile)

all_predictions_matched_2 = recommender_format(all_predictions_matched)

# Save matched file in second format
with open("predictions/" + name + "_matched_format_2.json", "w") as outfile:
    json.dump(all_predictions_matched_2, outfile)

