from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier

import dataset as dt
from sklearn import svm
import re
from score import evaluate_classifiers

feature_name = "HOG224"
train_features_path = r'C:\Users\emanu\OneDrive\Desktop\UniLM\Tesi\DATASET\{0}\dlib20\PMDB_cropped_20'.format(feature_name)
test_features_path = r'C:\Users\emanu\OneDrive\Desktop\UniLM\Tesi\DATASET\{0}\dlib20\MorphDB_cropped_20'.format(feature_name)

train_features = dt.load_feature_dataset(train_features_path)
test_features = dt.load_feature_dataset(test_features_path)

filename_train_filter = (lambda fullname: bool(re.search('morph.*0.55|.*.TestImages.*', fullname)))
filename_test_filter = (lambda fullname: bool(re.search('morph.*_D|.*.TestImages.*', fullname)))

print("Loading features...")
X_train, y_train = dt.filter_dataset(train_features, filename_filter=filename_train_filter)
X_test, y_test = dt.filter_dataset(test_features, filename_filter=filename_test_filter)
print("Features loaded...")


classifiers = [KNeighborsClassifier(n_neighbors=5), GaussianNB(), svm.SVC(probability=True), MLPClassifier(solver='lbfgs', alpha=1e-5,hidden_layer_sizes=(5, 2),random_state=1, max_iter=2000)]

evaluate_classifiers(X_train, y_train, X_test, y_test, feature_name, classifiers)
