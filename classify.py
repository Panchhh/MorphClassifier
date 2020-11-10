from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier

from dataset import single_image_dataset
from sklearn import svm
from score import evaluate_classifiers
from files import train_data_path, test_data_path, single_image_filename_train_filter, single_image_filename_test_filter

feature_name = "SIFT224_200"

print("Loading features...")
X_train, y_train = single_image_dataset(train_data_path(feature_name), data_type='feature',
                                        filename_filter=single_image_filename_train_filter)
X_test, y_test = single_image_dataset(test_data_path(feature_name), data_type='feature',
                                      filename_filter=single_image_filename_test_filter)
print("Features loaded...")


classifiers = [KNeighborsClassifier(n_neighbors=5), GaussianNB(), svm.SVC(probability=True), MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1, max_iter=2000)]

evaluate_classifiers(X_train, y_train, X_test, y_test, feature_name, classifiers)
