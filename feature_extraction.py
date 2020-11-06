import dataset as dt
from transformation import lbp_extractor, resizer, compose

input_root = r''
output_root = r''

#dt.transform_dataset(input_root, output_root, t.hog_extractor(8, (16, 16), (1, 1), True), "image", "feature")

dt.transform_dataset(input_root, output_root,
                     compose(lbp_extractor(1, 8, 'uniform'), resizer((224, 224))),
                     "image", "feature")
