import dataset as dt
from transformation import Sift
from files import transformation_root_path, train_data_path

input_name = "FM224"
output_name = "SIFT224_200"

train_images_path = train_data_path(input_name)
X_train, y_train = dt.single_image_dataset(train_images_path, data_type='image')

sift = Sift(X_train, 200)
dt.transform_dataset(transformation_root_path(input_name),
                     transformation_root_path(output_name),
                     sift.sift_extractor(), "image", "feature")

print("done")
