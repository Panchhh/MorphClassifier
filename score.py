import os
import time
from pathlib import Path
import numpy as np


def evaluate_classifiers(x_train, y_train, x_test, y_test, folder_name, classifiers):
    """
    Train and test the specified classifiers, saving the results.

    :param x_train: Training data.
    :param y_train: Training labels.
    :param x_test: Test data.
    :param y_test: Test labels.
    :param folder_name: Folder to save to (e.g. 'HOG_differential'). Actual path is 'scores\folder_name\'
    :param classifiers: List of classifiers. They must implement predict-proba (you may need to specify probability=True).
    """
    results = []
    for clf in classifiers:
        print("Training {0} with {1} elements".format(clf.__class__.__name__, y_train.size))
        start = time.clock()
        clf.fit(X=x_train, y=y_train)
        print("Training complete.")
        seconds = time.clock() - start
        results.append({
            "classifier": clf.__class__.__name__,
            "score": clf.score(x_test, y_test),
            "training_time_s": seconds,
        })
        _save_score(clf, folder_name, x_test, y_test)
    _save_file(r"scores\{0}\summary.score".format(folder_name), results)


def _save_score(clf, folder_name, X_test, y_test):
    print("Testing with {0} new elements".format(y_test.size))
    genuine_error_score = []
    impostor_success_score = []
    morph_class_index = np.where(clf.classes_ == "morphed")[0]
    for i, x in enumerate(X_test):
        morph_proba = clf.predict_proba(x.reshape(1, -1))[:, morph_class_index].item()
        if y_test[i] == "bonafide":
            genuine_error_score.append(morph_proba)
        else:
            impostor_success_score.append(morph_proba)

    _save_file(r"scores\{0}\{1}\{1}_bonafide.score".format(folder_name, clf.__class__.__name__), genuine_error_score)
    _save_file(r"scores\{0}\{1}\{1}_impostor.score".format(folder_name, clf.__class__.__name__), impostor_success_score)


def _save_file(filename, scores):
    Path(os.path.dirname(filename)).mkdir(parents=True, exist_ok=True)
    with open(filename, 'w') as file:
        for score in scores:
            file.write('{0}\n'.format(score))