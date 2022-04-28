
def assemble_predictions(predictions, x, sentences, max_len):
    all_outputs = dict()  # Stores all outputs from the test dataset per entity
    all_outputs_per_sentence = dict()  # Stores separate dictionaries per entity for every sentence in the dataset
    print(len(predictions))
    for i in range(0, len(predictions)):  # Sentences iteration
        tmp_dict = dict()
        for j in range(0, len(predictions[i]) - 2):  # Word iteration
            if predictions[i][j] == 'B-movie-pos':
                if 'positive_movies' not in all_outputs.keys():
                    all_outputs['positive_movies'] = []
                if 'positive_movies' not in tmp_dict.keys():
                    tmp_dict['positive_movies'] = []
                tmp_entity = x[i][j]
                k = j + 1
                while predictions[i][k] == 'I-movie-pos' and k < max_len - 1:
                    tmp_entity += ' ' + x[i][k]
                    k += 1
                all_outputs['positive_movies'].append(tmp_entity)
                if tmp_entity not in tmp_dict['positive_movies']:
                    tmp_dict['positive_movies'].append(tmp_entity)

            if predictions[i][j] == 'B-movie-neg':
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
                if 'positive_actors' not in all_outputs.keys():
                    all_outputs['positive_actors'] = []
                if 'positive_actors' not in tmp_dict.keys():
                    tmp_dict['positive_actors'] = []
                tmp_entity = x[i][j]
                k = j + 1
                while predictions[i][k] == 'I-actor-pos':
                    tmp_entity += ' ' + x[i][k]
                    k += 1
                all_outputs['positive_actors'].append(tmp_entity)
                if tmp_entity not in tmp_dict['positive_actors']:
                    tmp_dict['positive_actors'].append(tmp_entity)

            if predictions[i][j] == 'B-actor-neg':
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


