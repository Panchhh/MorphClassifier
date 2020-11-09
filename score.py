import time
import numpy as np


def evaluate_classifiers(x_train, y_train, x_test, y_test, feature_name, classifiers):
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
        save_score(clf, feature_name, x_test, y_test)
    save_file(r"scores\classifiers_{0}.score".format(feature_name), results)


def save_score(clf, feature_name, X_test, y_test):
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

    save_file(r"scores\{0}_{1}_bonafide.score".format(feature_name, clf.__class__.__name__), genuine_error_score)
    save_file(r"scores\{0}_{1}_impostor.score".format(feature_name, clf.__class__.__name__), impostor_success_score)


def save_file(filename, scores):
    with open(filename, 'w') as file:
        for score in scores:
            file.write('{0}\n'.format(score))
