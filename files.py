import re

_dataset_root = r'C:\Users\emanu\OneDrive\Desktop\UniLM\Tesi\DATASETS'
_differential_couples_root = r'C:\Users\emanu\OneDrive\Desktop\UniLM\Tesi\couples'


def transformation_root_path(feature_name):
    return r'{0}\{1}'.format(_dataset_root, feature_name)

######### PMDB #########
def train_data_path(feature_name):
    return r'{0}\{1}\dlib20\PMDB_cropped_20'.format(_dataset_root, feature_name)

# Single image
single_image_filename_train_filter = (lambda fullname: bool(re.search('morph.*0[.]55|.*.TestImages.*', fullname)))

# Differential
differential_train_couples_paths = [
    r"{0}\couples_pmdb_morphed_criminal_0.55.txt".format(_differential_couples_root),
    r"{0}\couples_pmdb_bona_fine.txt".format(_differential_couples_root),
]


######### MorphDB ##########
"""
def test_data_path(feature_name):
    return r'{0}\{1}\dlib20\MorphDB_cropped_20'.format(_dataset_root, feature_name)

# Single image MorphDB
single_image_filename_test_filter = (lambda fullname: bool(re.search('.*_D.*', fullname)))

# Differential MorphDB
differential_test_couples_paths = [
    r"{0}\couples_morphdb_bona_fide_ALL.txt".format(_differential_couples_root),
    r"{0}\couples_morphdb_morphed_criminal.txt".format(_differential_couples_root), #accomplice
]"""


######### LondonDB ##########
def test_data_path(feature_name):
    return r'{0}\{1}\inner'.format(_dataset_root, feature_name)

# Single image londonDB
single_image_filename_test_filter = (lambda fullname: bool(re.search('.*', fullname)))

# Differential londonDB
differential_test_couples_paths = [
    r"{0}\couples_londondb_bona_fide.txt".format(_differential_couples_root),
    r"{0}\couples_londondb_morphed_criminal.txt".format(_differential_couples_root), #accomplice
]
