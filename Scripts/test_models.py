
def flatten_predictions(predictions):
    """
    The labels of the sentences are usually
    returned per sentence in the form:
    [pred1, pred2,...],[pred1, pred2,...]
    For easier statistical evaluation,
    they are flattened to one list:
    [pred1, pred2, pred3,..]
    """
    flatten_preds = []
    for sentence in predictions:
        for tag in sentence:
            flatten_preds.append(tag)
    return flatten_preds
