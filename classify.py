from datetime import datetime
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
import dataset as dt
from sklearn import svm
import re

train_features_path = r''
test_features_path = r''

print("Loading features...")
train_features = dt.load_feature_dataset(train_features_path)
test_features = dt.load_feature_dataset(test_features_path)
print("Features loaded...")

filename_filter = (lambda fullname: bool(re.search('morph.*0.55|.*.TestImages.*', fullname)))

X_train, y_train = dt.filter_dataset(train_features, filename_filter=filename_filter)
X_test, y_test = dt.filter_dataset(test_features, filename_filter=filename_filter)


def evaluate_classifier(X_train, y_train, X_test, y_test, classifiers):
    results = []
    for clf in classifiers:
        print("Training {0} with {1} elements".format(clf.__class__.__name__, y_train.size))
        start = datetime.now()
        clf.fit(X=X_train, y=y_train)
        end = datetime.now() - start
        results.append({
            "classifier": clf.__class__.__name__,
            "score": clf.score(X_test, y_test),
            "training_time": end.microseconds
        })
    return results


classifiers = [KNeighborsClassifier(n_neighbors=5), GaussianNB(), svm.SVC(), MLPClassifier(solver='lbfgs', alpha=1e-5,
                                                                                           hidden_layer_sizes=(5, 2),
                                                                                           random_state=1,
                                                                                           max_iter=2000)]

print(evaluate_classifier(X_train, y_train, X_test, y_test, classifiers))
