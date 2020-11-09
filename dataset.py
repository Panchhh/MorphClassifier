import os
import cv2
from pathlib import Path
import numpy as np


def transform_dataset(input_path, output_path, function, input_data_type, output_data_type):
    """Applies a specified function to the entire dataset loaded from input_path and saves the results in the specified output_path keeping the same file system structure.

        :param str input_path: Input dataset root path.
        :param str output_path: Output transformed dataset root path.
        :param function: Transformation function.
        :param str input_data_type: Input data type. 'image' for images, 'feature' for numpy vectors
        :param str output_data_type: Output data type. 'image' for images, 'feature' for numpy vectors
        """
    for data in _dataset_iterator(input_path, input_data_type):
        data['output'] = function(data['data'])
        _save_data(output_path, data, 'output', output_data_type)


def single_image_dataset(path, data_type, data_filter=lambda x: True, filename_filter=lambda x: True):
    """
    A single image dataset loaded in memory. Filters on data properties or filename can be specified.

    :param path: Dataset root path.
    :param data_type: 'image' or 'feature'
    :param data_filter: Predicate on data properties (e.g. size). Identity function by default.
    :param filename_filter: Predicate on full_filename (e.g. regex). Identity function by default.

    :return: A pair of numpy array representing data and labels respectively.
    """
    dataset = _dataset_iterator(path, data_type)
    filtered_x, filtered_y = zip(
        *[(f['data'], f['class']) for f in dataset if (data_filter(f['data']) and filename_filter(f['full_filename']))])
    return np.array(filtered_x), np.array(filtered_y)


def differential_dataset(couples_files, dataset_folder, data_type, operation):
    """
    A differential dataset loaded in memory. Data pairs are taken from 'couples_files' list. Pairs are combined through the function 'operation' (e.g difference).

    :param couples_files: List of file
    :param dataset_folder: File list containing data pairs.
    :param data_type: 'image' or 'feature'
    :param operation: Data pairs combination function (data1, data2 => data) (e.g difference)

    :return: A pair of numpy array representing data and labels respectively.
    """
    dataset = _differential_dataset_iterator(couples_files, dataset_folder, data_type, operation)
    x, y = zip(*[(f['data'], f['class']) for f in dataset])
    return np.array(x), np.array(y)


def _dataset_iterator(path, data_type):
    root_path = path
    for filepath, _, filenames in os.walk(path):
        for filename in filenames:
            file = os.path.join(filepath, filename)
            yield {
                'path': filepath,
                'relative_path': os.path.relpath(filepath, root_path),
                'full_filename': file,
                'filename': filename,
                'data': _load_file(file, data_type),
                'class': _compute_class(filename)
            }


def _differential_dataset_iterator(couples_files, dataset_folder, data_type, operation):
    for couple_file in couples_files:
        with open(couple_file) as file:
            for row in file:
                ref, other = _parse_row(row, dataset_folder, data_type)
                yield {
                    'ref': ref,
                    'other': other,
                    'data': operation(_load_file(ref, data_type), _load_file(other, data_type)),
                    'class': _compute_class(os.path.basename(ref))
                }


def _parse_row(row, dataset_folder, data_type):
    def extract_path(col):
        filename_path = Path(os.path.join(dataset_folder, '\\'.join(col.split("\\")[2:])))
        if data_type == "image":
            filename_path = filename_path.with_suffix('.png')
        elif data_type == "feature":
            filename_path = filename_path.with_suffix('.npy')
        return filename_path

    cols = row.split("\t")
    return extract_path(cols[0]), extract_path(cols[1])


def _load_file(file, data_type):
    if data_type == 'image':
        return cv2.imread(file)
    elif data_type == 'feature':
        return np.load(file)


def _compute_class(base_filename):
    return 'morphed' if 'morph' in os.path.basename(base_filename) else 'bonafide'


def _save_data(path, data, data_field, data_type):
    filepath = os.path.join(path, data['relative_path'])
    Path(filepath).mkdir(parents=True, exist_ok=True)
    filename_path = Path(os.path.join(filepath, data['filename']))
    if data_type == 'image':
        filename_path = filename_path.with_suffix('.png')
        cv2.imwrite(str(filename_path), data[data_field])
    elif data_type == 'feature':
        filename_path = filename_path.with_suffix('.npy')
        np.save(filename_path, data[data_field])
