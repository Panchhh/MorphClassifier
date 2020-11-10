from sklearn.manifold import TSNE
from dataset import single_image_dataset
from matplotlib import pyplot as plt

from files import train_data_path, single_image_filename_train_filter, test_data_path, single_image_filename_test_filter

feature_name = "SIFT224_200"

print("Loading features...")
X_train, y_train = single_image_dataset(train_data_path(feature_name), data_type='feature',
                                        filename_filter=single_image_filename_train_filter)
X_test, y_test = single_image_dataset(test_data_path(feature_name), data_type='feature',
                                      filename_filter=single_image_filename_test_filter)
print("Features loaded...")


def plot_dataset(x, y):
    tsne = TSNE(n_components=2, random_state=0)
    feature = tsne.fit_transform(x)
    for i, f in enumerate(feature):
        if y[i] == "morphed":
            plt.scatter(feature[i, 0], feature[i, 1], c="red")
        else:
            plt.scatter(feature[i, 0], feature[i, 1], c="green")
    plt.show()


plot_dataset(X_train, y_train)
plot_dataset(X_test, y_test)
