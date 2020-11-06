import os
import cv2
from pathlib import Path
import numpy as np


def dataset_iterator(path, data_type):
    """A Dataset iterator. Iterate all files recursively starting from path.

        :param str path: Starting root path.
        :param str data_type: Data type. 'image' for images, 'feature' for numpy vectors
        :return: the dataset iterator. Each element it's a dictionary containing 'path', 'relative_path', 'filename', 'data', 'class' """
    root_path = path
    for filepath, _, filenames in os.walk(path):
        for filename in filenames:
            file = os.path.join(filepath, filename)
            yield {
                'path': filepath,
                'relative_path': os.path.relpath(filepath, root_path),
                'full_filename': file,
                'filename': filename,
                'data': cv2.imread(file) if data_type == 'image' else np.load(file),
                'class': ('morphed' if 'morph' in filename else 'bonafide')
            }


def filter_dataset(dataset, data_filter=lambda x: True, filename_filter=lambda x: True):
    """
    Filter dataset based on data and/or filename.
    :param dataset: Input dataset.
    :param data_filter: Filter to apply on data. (e.g. data dimensionality)
    :param filename_filter: Filter to apply on filename.
    :return: a tuple containing data and label vectors.
    """
    # TODO: maybe filter in a lazy way but data must be loaded in order to train classifier
    filtered_x, filtered_y = zip(
        *[(f['data'], f['class']) for f in dataset if (data_filter(f['data']) and filename_filter(f['full_filename']))])
    return np.array(filtered_x), np.array(filtered_y)


def load_image_dataset(path):
    """Load an image dataset in memory.

        :param str path: Starting root path.
        :return: the dataset. Each element it's a dictionary containing 'path', 'relative_path', 'filename', 'data', 'class' """
    return load_dataset(path, data_type='image')


def load_feature_dataset(path):
    """Load a feature dataset in memory.

        :param str path: Starting root path.
        :return: the dataset. Each element it's a dictionary containing 'path', 'relative_path', 'filename', 'data', 'class' """
    return load_dataset(path, data_type='feature')


def load_dataset(path, data_type):
    dataset = []
    for data in dataset_iterator(path, data_type):
        dataset.append(data)
    return dataset


def save_data(path, data, data_field, data_type):
    filepath = os.path.join(path, data['relative_path'])
    Path(filepath).mkdir(parents=True, exist_ok=True)
    filename_path = Path(os.path.join(filepath, data['filename']))
    if data_type == 'image':
        filename_path = filename_path.with_suffix('.png')
        cv2.imwrite(str(filename_path), data[data_field])
    elif data_type == 'feature':
        filename_path = filename_path.with_suffix('.npy')
        np.save(filename_path, data[data_field])


def transform_dataset(input_path, output_path, function, input_data_type, output_data_type):
    """Applies a specified function to the entire dataset loaded from input_path and saves the results in the specified output_path keeping the same file system structure.

        :param str input_path: Input dataset root path.
        :param str output_path: Output transformed dataset root path.
        :param function: Transformation function.
        :param str input_data_type: Input data type. 'image' for images, 'feature' for numpy vectors
        :param str output_data_type: Output data type. 'image' for images, 'feature' for numpy vectors
        """
    for data in dataset_iterator(input_path, input_data_type):
        data['output'] = function(data['data'])
        save_data(output_path, data, 'output', output_data_type)
