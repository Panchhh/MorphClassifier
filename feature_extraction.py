import dataset as dt
from transformation import lbp_extractor, resizer, compose, hog_extractor

input_root = r'C:\Users\emanu\OneDrive\Desktop\UniLM\Tesi\DATASET\FM224'
output_root = r'C:\Users\emanu\OneDrive\Desktop\UniLM\Tesi\DATASET\HOG224_32'

#dt.transform_dataset(input_root, output_root, hog_extractor(8, (32, 32), (1, 1), True), "image", "feature")

dt.transform_dataset(input_root, output_root,
                     compose(lbp_extractor(1, 8, 'uniform'), resizer((224, 224))),
                     "image", "feature")