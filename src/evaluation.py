import pandas as pd
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay

def evaluate_model(model, X_test, y_test, name):

    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_test, y_pred, average='macro'
    )


    return [name, acc, precision, recall, f1]


def evaluate_all(models, X_test, y_test):
    results = []
    for name, model in models.items():
        results.append(evaluate_model(model, X_test, y_test, name))

    results_df = pd.DataFrame(
        results,
        columns=["Model", "Accuracy", "Precision", "Recall", "F1-score"]
    )

    return results_df
