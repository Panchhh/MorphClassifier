from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier

from dataset import differential_dataset
from sklearn import svm
from score import evaluate_classifiers

feature_name = "HOG224"
train_features_path = r'C:\Users\emanu\OneDrive\Desktop\UniLM\Tesi\DATASET\{0}\dlib20\PMDB_cropped_20'.format(
    feature_name)
test_features_path = r'C:\Users\emanu\OneDrive\Desktop\UniLM\Tesi\DATASET\{0}\dlib20\MorphDB_cropped_20'.format(
    feature_name)

train_couples_paths = [
    r"C:\Users\emanu\OneDrive\Desktop\UniLM\Tesi\couples\couples_pmdb_morphed_accomplice_0.55.txt",
    r"C:\Users\emanu\OneDrive\Desktop\UniLM\Tesi\couples\couples_pmdb_bona_fine.txt",
]

test_couples_paths = [
    r"C:\Users\emanu\OneDrive\Desktop\UniLM\Tesi\couples\couples_morphdb_bona_fide_ALL.txt",
    r"C:\Users\emanu\OneDrive\Desktop\UniLM\Tesi\couples\couples_morphdb_morphed_criminal.txt",
]


def difference(ref, other):
    return ref - other


print("Loading features...")
X_train, y_train = differential_dataset(train_couples_paths, train_features_path, "feature", difference)
X_test, y_test = differential_dataset(test_couples_paths, test_features_path, "feature", difference)
print("Features loaded...")

classifiers = [KNeighborsClassifier(n_neighbors=5), GaussianNB(), svm.SVC(probability=True),
               MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1, max_iter=2000)]

evaluate_classifiers(X_train, y_train, X_test, y_test, feature_name + "_differential", classifiers)
