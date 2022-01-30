

def evaluate_test_data(model, X_test, y_test):
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