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
                'filename': filename,
                'data': cv2.imread(file) if data_type == 'image' else np.load(file),
                'class': ('morphed' if 'morph' in filename else 'bonafide')
            }


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
    filename = Path(os.path.join(filepath, data['filename']))
    if data_type == 'image':
        filename = filename.with_suffix('.png')
        cv2.imwrite(filename, data['data'])
    elif data_type == 'feature':
        filename = filename.with_suffix('.npy')
        np.save(filename, data[data_field])


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
