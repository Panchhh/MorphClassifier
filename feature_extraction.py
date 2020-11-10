import dataset as dt
from transformation import lbp_extractor, resizer, compose, hog_extractor
from files import transformation_root_path

input_root = transformation_root_path("FM224")
output_root = transformation_root_path("HOG224")

#dt.transform_dataset(input_root, output_root, hog_extractor(8, (32, 32), (1, 1), True), "image", "feature")

dt.transform_dataset(input_root, output_root,
                     compose(lbp_extractor(1, 8, 'uniform'), resizer((224, 224))),
                     "image", "feature")
