import numpy as np


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


def evaluate_from_model(model, X_test, y_test):
    """
    Accepts a trained model, test set and list
    of original labels. Returns a classification
    report with precision, recall and f1 scores.
    """

    p = model.predict(np.array(X_test))
    p = np.argmax(p, axis=-1)
    y_test = np.array(y_test)
    y_test = np.argmax(y_test, axis=-1)

    y_orig = []
    for sent in y_test:
        for tag in sent:
            y_orig.append(tag)

    y_preds = []
    for sent in p:
        for tag in sent:
            y_preds.append(tag)

    report = classification_report(y_orig, y_preds)
    print(report)