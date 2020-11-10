from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier

from dataset import differential_dataset
from sklearn import svm
from score import evaluate_classifiers
from files import train_data_path, test_data_path, differential_train_couples_paths, differential_test_couples_paths

feature_name = "SIFT224_200"


def difference(ref, other):
    return ref - other


print("Loading features...")
X_train, y_train = differential_dataset(differential_train_couples_paths, train_data_path(feature_name), "feature", difference)
X_test, y_test = differential_dataset(differential_test_couples_paths, test_data_path(feature_name), "feature", difference)
print("Features loaded...")

classifiers = [KNeighborsClassifier(n_neighbors=5), GaussianNB(), svm.SVC(probability=True),
               MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1, max_iter=2000)]

evaluate_classifiers(X_train, y_train, X_test, y_test, feature_name + "_differential", classifiers)
