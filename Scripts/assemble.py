from difflib import SequenceMatcher


def similar(a, b):
    return SequenceMatcher(None, a, b).ratio()


def assemble_predictions(predictions, x, sentences, name):
    all_outputs = dict()  # Stores all outputs from the test dataset per entity
    all_outputs_per_sentence = dict()  # Stores separate dictionaries per entity for every sentence in the dataset
    for i in range(0, len(predictions)):  # Sentences iteration
        tmp_dict = dict()
        for j in range(0, len(predictions[i]) - 2):  # Word iteration
            if predictions[i][j] == 'B-movie-pos':
                if name == 'elmo':
                    if 'positive_movies' not in all_outputs.keys():
                        all_outputs['positive_movies'] = []
                    if 'positive_movies' not in tmp_dict.keys():
                        tmp_dict['positive_movies'] = []
                    tmp_entity = x[i][j]
                    k = j + 1
                    while predictions[i][k] == 'I-movie-pos' and k < len(predictions[i]) - 1:
                        tmp_entity += ' ' + x[i][k]
                        k += 1
                    all_outputs['positive_movies'].append(tmp_entity)
                    if tmp_entity not in tmp_dict['positive_movies']:
                        tmp_dict['positive_movies'].append(tmp_entity)

            if predictions[i][j] == 'B-movie-neg':
                if name == 'roberta':
                    if 'negative_movies' not in all_outputs.keys():
                        all_outputs['negative_movies'] = []
                    if 'negative_movies' not in tmp_dict.keys():
                        tmp_dict['negative_movies'] = []
                    tmp_entity = x[i][j]
                    k = j + 1
                    while predictions[i][k] == 'I-movie-neg':
                        tmp_entity += ' ' + x[i][k]
                        k += 1
                    all_outputs['negative_movies'].append(tmp_entity)
                    if tmp_entity not in tmp_dict['negative_movies']:
                        tmp_dict['negative_movies'].append(tmp_entity)

            if predictions[i][j] == 'B-keyword-pos':
                if name == 'elmo':
                    if 'positive_keywords' not in all_outputs.keys():
                        all_outputs['positive_keywords'] = []
                    if 'positive_keywords' not in tmp_dict.keys():
                        tmp_dict['positive_keywords'] = []
                    tmp_entity = x[i][j]
                    k = j + 1
                    while predictions[i][k] == 'I-keyword-pos':
                        tmp_entity += ' ' + x[i][k]
                        k += 1
                    all_outputs['positive_keywords'].append(tmp_entity)
                    if tmp_entity not in tmp_dict['positive_keywords']:
                        tmp_dict['positive_keywords'].append(tmp_entity)

            if predictions[i][j] == 'B-keyword-neg':
                if name == 'roberta':
                    if 'negative_keywords' not in all_outputs.keys():
                        all_outputs['negative_keywords'] = []
                    if 'negative_keywords' not in tmp_dict.keys():
                        tmp_dict['negative_keywords'] = []
                    tmp_entity = x[i][j]
                    k = j + 1
                    while predictions[i][k] == 'I-keyword-neg':
                        tmp_entity += ' ' + x[i][k]
                        k += 1
                    all_outputs['negative_keywords'].append(tmp_entity)
                    if tmp_entity not in tmp_dict['negative_keywords']:
                        tmp_dict['negative_keywords'].append(tmp_entity)

            if predictions[i][j] == 'B-actor-pos':
                if name == 'bert_mult':
                    if 'positive_actors' not in all_outputs.keys():
                        all_outputs['positive_actors'] = []
                    if 'positive_actors' not in tmp_dict.keys():
                        tmp_dict['positive_actors'] = []
                    tmp_entity = x[i][j]
                    k = j + 1
                    while predictions[i][k] == 'I-actor-pos' and k < len(predictions[i]) - 1:
                        tmp_entity += ' ' + x[i][k]
                        k += 1
                    all_outputs['positive_actors'].append(tmp_entity)
                    if tmp_entity not in tmp_dict['positive_actors']:
                        tmp_dict['positive_actors'].append(tmp_entity)

            if predictions[i][j] == 'B-actor-neg':
                if name == 'roberta':
                    if 'negative_actors' not in all_outputs.keys():
                        all_outputs['negative_actors'] = []
                    if 'negative_actors' not in tmp_dict.keys():
                        tmp_dict['negative_actors'] = []
                    tmp_entity = x[i][j]
                    k = j + 1
                    while predictions[i][k] == 'I-actor-neg':
                        tmp_entity += ' ' + x[i][k]
                        k += 1
                    all_outputs['negative_actors'].append(tmp_entity)
                    if tmp_entity not in tmp_dict['negative_actors']:
                        tmp_dict['negative_actors'].append(tmp_entity)

            if predictions[i][j] == 'B-gen-pos':
                if name == 'elmo':
                    if 'positive_genres' not in all_outputs.keys():
                        all_outputs['positive_genres'] = []
                    if 'positive_genres' not in tmp_dict.keys():
                        tmp_dict['positive_genres'] = []
                    tmp_entity = x[i][j]
                    k = j + 1
                    while predictions[i][k] == 'I-gen-pos':
                        tmp_entity += ' ' + x[i][k]
                        k += 1
                    all_outputs['positive_genres'].append(tmp_entity)
                    if tmp_entity not in tmp_dict['positive_genres']:
                        tmp_dict['positive_genres'].append(tmp_entity)

            if predictions[i][j] == 'B-gen-neg':
                if name == 'bert_mult':
                    if 'negative_genres' not in all_outputs.keys():
                        all_outputs['negative_genres'] = []
                    if 'negative_genres' not in tmp_dict.keys():
                        tmp_dict['negative_genres'] = []
                    tmp_entity = x[i][j]
                    k = j + 1
                    while predictions[i][k] == 'I-gen-neg':
                        tmp_entity += ' ' + x[i][k]
                        k += 1
                    all_outputs['negative_genres'].append(tmp_entity)
                    if tmp_entity not in tmp_dict['negative_genres']:
                        tmp_dict['negative_genres'].append(tmp_entity)

        if i < len(sentences):
            all_outputs_per_sentence[sentences[i][0][0]] = tmp_dict

    return all_outputs, all_outputs_per_sentence


def split_keyphrases(dict):
    for key in dict.keys():
        if "positive_keywords" in dict[key].keys():
            tmp_pos_keys = []
            for keyphrase in dict[key]["positive_keywords"]:
                keywords = keyphrase.split(" ")
                tmp_pos_keys.extend(keywords)
            dict[key]["positive_keywords"] = list(set(tmp_pos_keys))
        if "negative_keywords" in dict[key].keys():
            tmp_neg_keys = []
            for keyphrase in dict[key]["negative_keywords"]:
                keywords = keyphrase.split(" ")
                tmp_neg_keys.extend(keywords)
            dict[key]["negative_keywords"] = list(set(tmp_neg_keys))
    return dict


def fix_prediction_dicts(all_predictions):
    """Fix completely wrong predictions, where 'NEW LINE' was returned
    as a movie. Accepts a dictionary of predictions where the keys are
    the submission ids."""

    for key in all_predictions:
        if 'positive_movies' in all_predictions[key].keys():
            all_predictions[key]["positive_movies"] = [item for item in all_predictions[key]["positive_movies"] if
                                                       item not in ['/', "The", 'of' "NEW_LINE", '.']]
            if len(all_predictions[key]["positive_movies"]) < 1:
                all_predictions[key].pop('positive_movies')

        if 'negative_movies' in all_predictions[key].keys():
            all_predictions[key]["negative_movies"] = [item for item in all_predictions[key]["negative_movies"] if
                                                       item not in ["/", "The", 'of', "NEW_LINE", "."]]
            if len(all_predictions[key]["negative_movies"]) < 1:
                all_predictions[key].pop('negative_movies')

        if 'positive_keywords' in all_predictions[key].keys():
            all_predictions[key]["positive_keywords"] = [item for item in all_predictions[key]["positive_keywords"] if
                                                         item not in ["/", "NEW_LINE", "."]]
            if len(all_predictions[key]["positive_keywords"]) < 1:
                all_predictions[key].pop('positive_keywords')

        if 'negative_keywords' in all_predictions[key].keys():
            all_predictions[key]["negative_keywords"] = [item for item in all_predictions[key]["negative_keywords"] if
                                                         item not in ["/", "NEW_LINE", "."]]
            if len(all_predictions[key]["negative_keywords"]) < 1:
                all_predictions[key].pop('negativee_keywords')

        if 'positive_keywords' in all_predictions[key].keys():
            for i in range(0, len(all_predictions[key]["positive_keywords"])):
                all_predictions[key]["positive_keywords"][i] = all_predictions[key]["positive_keywords"][i].replace("/",
                                                                                                                    "")

        if 'negative_keywords' in all_predictions[key].keys():
            for i in range(0, len(all_predictions[key]["negative_keywords"])):
                all_predictions[key]["negative_keywords"][i] = all_predictions[key]["negative_keywords"][i].replace("/",
                                                                                                                    "")
    return all_predictions


def recommender_format(all_predictions):
    """Change the format of the predictions to fit
    recommender engine"""

    all_predictions_2 = dict()

    for key, value in all_predictions.items():
        if not key in all_predictions_2.keys():
            all_predictions_2[key] = list()
            for k in value.keys():
                if k == 'positive_movies':
                    for v in value[k]:
                        all_predictions_2[key].append("-mp " + "\"" + v + "\"")
                elif k == 'negative_movies':
                    for v in value[k]:
                        all_predictions_2[key].append("-mn " + "\"" + v + "\"")
                elif k == 'positive_genres':
                    for v in value[k]:
                        all_predictions_2[key].append("-gp " + "\"" + v + "\"")
                elif k == 'negative_genres':
                    for v in value[k]:
                        all_predictions_2[key].append("-gn " + "\"" + v + "\"")
                elif k == 'positive_actors':
                    for v in value[k]:
                        all_predictions_2[key].append("-amp " + "\"" + v + "\"")
                elif k == 'positive_actors':
                    for v in value[k]:
                        all_predictions_2[key].append("-amn " + "\"" + v + "\"")
                elif k == 'positive_keywords':
                    for v in value[k]:
                        all_predictions_2[key].append("-kp " + "\"" + v + "\"")
                elif k == 'negative_keywords':
                    for v in value[k]:
                        all_predictions_2[key].append("-kn " + "\"" + v + "\"")
    return all_predictions_2


def do_matching(all_predictions, imdb_genres, movie_titles, alt_names):

    imdb_predictions = dict()
    for key in all_predictions.keys():
        # print(all_predictions[key])
        imdb_predictions[key] = dict()
        for category in all_predictions[key].keys():
            if category not in imdb_predictions[key].keys():
                imdb_predictions[key][category] = []

            if category in ['positive_keywords', 'negative_keywords']:
                imdb_predictions[key][category] = all_predictions[key][category]

            elif category in ['positive_genres', 'negative_genres']:
                for genre in all_predictions[key][category]:
                    for genre_imdb in imdb_genres.genrename.values:
                        if genre == genre_imdb:
                            imdb_predictions[key][category].append(genre_imdb)
                        else:
                            if similar(genre, genre_imdb) > 0.70:
                                imdb_predictions[key][category].append(genre_imdb)

                all_predictions[key][category] = list(set(all_predictions[key][category]))

            elif category in ['positive_actors', 'negative_actors']:
                imdb_predictions[key][category] = all_predictions[key][category]

            elif category in ['positive_movies', 'negative_movies']:
                for movie in all_predictions[key][category]:
                    most_similar = [mov for mov in movie_titles.movie_title.values if
                                    ((similar(movie, mov[0:-7]) >= 0.9) or similar('The ' + movie, mov[0:-7]) >= 0.9)]
                    # most_similar = [mov for mov in movie_titles.movie_title.values if ((similar(movie, mov[0:len(movie)]) >= 0.8) or similar('The' + movie, mov[0:len(movie)+3]) >= 0.8)]

                    if len(most_similar) > 0:
                        most_similar_id = [movie_titles[movie_titles["movie_title"] == title]["movie_id"].values[0] for
                                           title in most_similar]
                        most_similar_id = list(dict.fromkeys(most_similar_id))
                        # print(most_similar)
                        # print(most_similar_id)
                        imdb_predictions[key][category].extend(most_similar_id)
                    else:
                        most_similar_alt = []
                        for k in alt_names.keys():
                            for mov in alt_names[k]:
                                if similar(movie, mov[0:-7]) >= 0.8:
                                    most_similar_alt.append(k)
                                    most_similar_alt.append(mov)

                        most_similar_alt = list(dict.fromkeys(most_similar_alt))
                        if len(most_similar_alt) > 0:
                            most_similar_alt_id = []
                            for title in most_similar_alt:
                                if title in movie_titles.movie_title.values:
                                    most_similar_alt_id.append(
                                        movie_titles[movie_titles["movie_title"] == title]["movie_id"].values[0])
                            most_similar_alt_id = list(dict.fromkeys(most_similar_alt_id))
                            imdb_predictions[key][category].extend(most_similar_alt_id)

            #        for movie_database in movie_titles.movie_title

    return imdb_predictions


def ensemble_appr(roberta, bert_multilingual, elmo):
    all_models = dict()
    for submission in roberta.keys():
        all_models[submission] = dict()
        if 'positive_movies' in elmo[submission].keys():
            all_models[submission]['positive_movies'] = elmo[submission]['positive_movies']
        if 'negative_movies' in roberta[submission].keys():
            all_models[submission]['negative_movies'] = roberta[submission]['negative_movies']
        if 'positive_genres' in elmo[submission].keys():
            all_models[submission]['positive_genres'] = elmo[submission]['positive_genres']
        if 'negative_genres' in bert_multilingual[submission].keys():
            all_models[submission]['negative_genres'] = bert_multilingual[submission]['negative_genres']
        # if 'negative_genres' in bert_large[submission].keys():
        #    all_models[submission]['negative_genres'] = bert_large[submission]['negative_genres']
        if 'positive_keywords' in elmo[submission].keys():
            all_models[submission]['positive_keywords'] = elmo[submission]['positive_keywords']
        if 'negative_keywords' in roberta[submission].keys():
            all_models[submission]['negative_keywords'] = roberta[submission]['negative_keywords']
        # if 'negative_keywords' in bert_large[submission].keys():
        #    all_models[submission]['negative_keywords'] = bert_large[submission]['negative_keywords']
        if 'positive_actors' in bert_multilingual[submission].keys():
            all_models[submission]['positive_actors'] = bert_multilingual[submission]['positive_actors']
        if 'negative_actors' in roberta[submission].keys():
            all_models[submission]['negative_actors'] = roberta[submission]['negative_actors']
    return all_models

