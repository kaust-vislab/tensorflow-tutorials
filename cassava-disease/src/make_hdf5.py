import glob
from PIL import Image

import h5py
import numpy as np

VERBOSE = True

with h5py.File("../data/interim/cassava-disease.hdf5") as hdf:

    testing_group = hdf.create_group("test/0")
    training_group = hdf.create_group("train")
    extra_images_group = hdf.create_group("extraimages")

    # write the test images into the hdf5 file 
    test_image_files = glob.glob("../data/raw/test/0/test-img-*.jpg")
    N = len(test_image_files)
    for _i, _file in enumerate(test_image_files):
        _image = Image.open(_file)
        _arr = np.asarray(_image)
        _name = _file.split('/')[-1]
        _ = testing_group.create_dataset(_name, data=_arr, compression="gzip")
        if _i % 10 == 0 and VERBOSE:
            print(f"Done writing {_i} out of {N} test images into the HDF5 file...")
    print("Done writing test images!\n")

    # write the training images into the hdf5 file
    classes = ["cbb", "cbsd", "cgm", "cmd", "healthy"]
    for _class in classes:
        _train_image_files = glob.glob(f"../data/raw/train/{_class}/train-{_class}-*.jpg")
        _N = len(_train_image_files)
        for _i, _file in enumerate(_train_image_files):
            _image = Image.open(_file)
            _arr = np.asarray(_image)
            _name = _file.split('/')[-1]
            _dataset = training_group.create_dataset(_name, data=_arr, compression="gzip")
            _dataset.attrs['class'] = _class
            if _i % 10 == 0 and VERBOSE:
                print(f"Done writing {_i} out of {_N} train images for class {_class} into the HDF5 file...")
        print(f"Done writing train images for class {_class}!\n")
    print("Done writing train images!\n")

    # write the extra images into the hdf5 file 
    extra_image_files = glob.glob("../data/raw/extraimages/extra-image-*.jpg")
    N = len(extra_image_files)
    for _i, _file in enumerate(extra_image_files):
        _image = Image.open(_file)
        _arr = np.asarray(_image)
        _name = _file.split('/')[-1]
        _ = extra_images_group.create_dataset(_name, data=_arr, compression="gzip")
        if _i % 10 == 0 and VERBOSE:
            print(f"Done writing {_i} out of {N} extra images into the HDF5 file...")
    print("Done writing extra images!\n")
