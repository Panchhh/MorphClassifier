import dataset as dt
from skimage.feature import hog

input_root = r''
output_root = r''


def hog_extraction(data):
    return hog(data, orientations=8, pixels_per_cell=(16, 16), cells_per_block=(1, 1), multichannel=True)


dt.transform_dataset(input_root, output_root, hog_extraction, "image", "feature")
