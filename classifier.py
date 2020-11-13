import itertools
import time

from sklearn import svm
from dataset import single_image_datasets, differential_datasets
from files import train_data_path, single_image_filename_train_filter, test_data_path, \
    single_image_filename_test_filter, differential_train_couples_paths, differential_test_couples_paths
from score import evaluate_classifiers

classifiers = [svm.SVC(probability=True, kernel='rbf')]

features_names_224 = [
    "HOG224",
    "LBP224",
    "LBP224_25",
    "LBPH224",
    "LBPH224_25",
    "SIFT224_200",
    "SURF224_200",
    "SIFT224_500",
    "SURF224_500"
]

features_names_320 = [
    "HOG320",
    "LBP320",
    "LBP320_25",
    "LBPH320",
    "LBPH320_25",
    "SIFT320_200",
    "SURF320_200",
]

operations = [
    ("subtraction", lambda ref, other: ref - other)
]


def all_combinations(elements, min_group_size, max_group_size):
    for L in range(min_group_size, max_group_size + 1):
        for subset in itertools.combinations(elements, L):
            yield list(subset)


def single_image_evaluate_feature(feature_names, classifiers):
    train_features_path = list(map(lambda f: train_data_path(f), feature_names))
    test_features_path = list(map(lambda f: test_data_path(f), feature_names))
    x_train, y_train = single_image_datasets(train_features_path, data_type='feature',
                                             filename_filter=single_image_filename_train_filter)
    x_test, y_test = single_image_datasets(test_features_path, data_type='feature',
                                           filename_filter=single_image_filename_test_filter)
    evaluate_classifiers(x_train, y_train, x_test, y_test, "+".join(feature_names), classifiers)


def differential_image_evaluate_feature(feature_names, classifiers, operations):
    train_features_path = list(map(lambda f: train_data_path(f), feature_names))
    test_features_path = list(map(lambda f: test_data_path(f), feature_names))
    for op_name, op in operations:
        x_train, y_train = differential_datasets(differential_train_couples_paths, train_features_path,
                                                 "feature", op)
        x_test, y_test = differential_datasets(differential_test_couples_paths, test_features_path, "feature",
                                               op)
        evaluate_classifiers(x_train, y_train, x_test, y_test, "{0}_diff_{1}".format("+".join(feature_names), op_name),
                             classifiers)


def benchmark(classifiers, feature_names, operations):
    start = time.clock()
    for features in feature_names:
        print("Classifying {0} features.... ".format(features), flush=True)
        single_image_evaluate_feature(features, classifiers)
        differential_image_evaluate_feature(features, classifiers, operations)
    print("Benchmark completed in {0} seconds".format(time.clock() - start))


benchmark(classifiers, [*all_combinations(features_names_224, 1,
                                          len(features_names_224))] + [
              *all_combinations(features_names_320, 1, len(features_names_320))], operations)
