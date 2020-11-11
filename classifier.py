import time

from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier

from dataset import single_image_dataset, differential_dataset
from files import train_data_path, single_image_filename_train_filter, test_data_path, \
    single_image_filename_test_filter, differential_train_couples_paths, differential_test_couples_paths
from score import evaluate_classifiers

classifiers = [KNeighborsClassifier(n_neighbors=10),
               GaussianNB(),
               svm.SVC(probability=True, kernel='rbf'),
               RandomForestClassifier(),
               MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=0, max_iter=1000)
               ]

feature_names = [
    "HOG224",
    "HOG320",
    "SIFT224_200",
    "SIFT320_200",
    "SURF224_200",
    "SURF320_200",
    "LBP224",
    "LBP320"
]

operations = [
    ("subtraction", lambda ref, other: ref - other)
]


def single_image_evaluate_feature(feature_name, classifiers):
    X_train, y_train = single_image_dataset(train_data_path(feature_name), data_type='feature',
                                            filename_filter=single_image_filename_train_filter)
    X_test, y_test = single_image_dataset(test_data_path(feature_name), data_type='feature',
                                          filename_filter=single_image_filename_test_filter)
    evaluate_classifiers(X_train, y_train, X_test, y_test, feature_name, classifiers)


def differential_image_evaluate_feature(feature_name, classifiers, operations):
    for op_name, op in operations:
        X_train, y_train = differential_dataset(differential_train_couples_paths, train_data_path(feature_name),
                                                "feature", op)
        X_test, y_test = differential_dataset(differential_test_couples_paths, test_data_path(feature_name), "feature",
                                              op)
        evaluate_classifiers(X_train, y_train, X_test, y_test, "{0}_diff_{1}".format(feature_name, op_name),
                             classifiers)


def benchmark(classifiers, feature_names, operations):
    start = time.clock()
    for feature_name in feature_names:
        print("Classifying {0} features.... ".format(feature_name), flush=True)
        single_image_evaluate_feature(feature_name, classifiers)
        differential_image_evaluate_feature(feature_name, classifiers, operations)
    print("Benchmark completed in {0} seconds".format(time.clock() - start))


benchmark(classifiers, feature_names, operations)
