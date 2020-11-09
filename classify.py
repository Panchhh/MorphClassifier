from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier

from dataset import single_image_dataset
from sklearn import svm
import re
from score import evaluate_classifiers

feature_name = "HOG224"
train_features_path = r'C:\Users\emanu\OneDrive\Desktop\UniLM\Tesi\DATASET\{0}\dlib20\PMDB_cropped_20'.format(feature_name)
test_features_path = r'C:\Users\emanu\OneDrive\Desktop\UniLM\Tesi\DATASET\{0}\dlib20\MorphDB_cropped_20'.format(feature_name)

filename_train_filter = (lambda fullname: bool(re.search('morph.*0.55|.*.TestImages.*', fullname)))
filename_test_filter = (lambda fullname: bool(re.search('.*_D.*', fullname)))

print("Loading features...")
X_train, y_train = single_image_dataset(train_features_path, data_type='feature', filename_filter=filename_train_filter)
X_test, y_test = single_image_dataset(test_features_path, data_type='feature', filename_filter=filename_test_filter)
print("Features loaded...")


classifiers = [KNeighborsClassifier(n_neighbors=5), GaussianNB(), svm.SVC(probability=True), MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1, max_iter=2000)]

evaluate_classifiers(X_train, y_train, X_test, y_test, feature_name, classifiers)
