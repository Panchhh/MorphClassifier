import os
import time

from dataset import transform_dataset, single_image_dataset
from files import transformation_root_path, train_data_path
from transformation import lbp_extractor, resizer, Sift, Surf, hog_extractor, lbp_hist_extractor

lbp_default = lambda: lbp_extractor(1, 8, 'uniform')
lbp_big = lambda: lbp_extractor(3, 24, 'uniform')
hog_default = lambda: hog_extractor(8, (16, 16), (1, 1), True)

lbp_hist_default = lambda: lbp_hist_extractor(1, 8, 'uniform')
lbp_hist_big = lambda: lbp_hist_extractor(3, 24, 'uniform')


def sift(feature_name, dimensionality):
    return lambda: Sift(single_image_dataset(train_data_path(feature_name), data_type='image')[0],
                        dimensionality).feature_extractor()


def surf(feature_name, dimensionality):
    return lambda: Surf(single_image_dataset(train_data_path(feature_name), data_type='image')[0],
                        dimensionality).feature_extractor()


simple_transformations = [
    {
        "semantic": "Image manipulation",
        "transformations": [
            ("FM", lambda: (resizer((320, 320))), "FM320"),
            ("FM", lambda: (resizer((224, 224))), "FM224")
        ],
        "types": ("image", "image")
    },
    {
        "semantic": "LBP",
        "transformations": [
            ("FM224", lbp_default, "LBP224"),
            ("FM320", lbp_default, "LBP320"),
            ("FM224", lbp_big, "LBP224_25"),
            ("FM320", lbp_big, "LBP320_25"),
            ("FM224", lbp_hist_default, "LBPH224"),
            ("FM320", lbp_hist_default, "LBPH320"),
            ("FM224", lbp_hist_big, "LBPH224_25"),
            ("FM320", lbp_hist_big, "LBPH320_25")
        ],
        "types": ("image", "feature")
    },
    {
        "semantic": "HOG",
        "transformations": [
            ("FM224", hog_default, "HOG224"),
            ("FM320", hog_default, "HOG320")
        ],
        "types": ("image", "feature")
    }
]

bag_of_word_transformation = [
    {
        "semantic": "SIFT",
        "transformations": [
            ("FM224", sift("FM224", 200), "SIFT224_200"),
            ("FM224", sift("FM224", 500), "SIFT224_500"),
            ("FM320", sift("FM320", 200), "SIFT320_200"),
        ],
        "types": ("image", "feature")
    },
    {
        "semantic": "SURF",
        "transformations": [
            ("FM224", surf("FM224", 200), "SURF224_200"),
            ("FM224", surf("FM224", 500), "SURF224_500"),
            ("FM320", surf("FM320", 200), "SURF320_200")
        ],
        "types": ("image", "feature")
    }
]


def datasets_generator(transformations):
    start = time.clock()

    for t_data in transformations:
        for input_folder, lazy_transformation, output_folder in t_data["transformations"]:
            input_path = transformation_root_path(input_folder)
            output_path = transformation_root_path(output_folder)
            print("Transformation {0} => {1}.... ".format(input_path, output_path), end="", flush=True)
            if not os.path.isdir(output_path):
                transform_dataset(input_path, output_path, lazy_transformation(), t_data["types"][0],
                                  t_data["types"][1])
                print("Done.", flush=True)
            else:
                print("Skipped.", flush=True)
    print("Generation completed in {0} seconds".format(time.clock() - start))


datasets_generator(simple_transformations + bag_of_word_transformation)
