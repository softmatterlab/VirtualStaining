from apido.deeptrack.features import Lambda
import os
import re
import glob
import random
import itertools
import apido
import numpy as np

from apido import deeptrack as dt

_file_name_struct = "HepaRG_{0}_S{1}.tif"

conf = {}


def DataLoader(
        path=None,
        dataset="HepaRG",
        **kwargs):

    DATASET_PATH = os.path.join(path, dataset)

    TRAINING_PATH = os.path.join(DATASET_PATH, "training_data")
    VALIDATION_PATH = os.path.join(DATASET_PATH, "validation_data")

    training_set = set([file.split('_')[-1][1:3]
                        for file in os.listdir(TRAINING_PATH)])

    validation_set = set([file.split('_')[-1][1:3]
                          for file in os.listdir(VALIDATION_PATH)])

    print("Training on {0} images".format(len(training_set)))
    print("Validating on {0} images".format(len(validation_set)))

    training_iterator = itertools.cycle(training_set)
    validation_iterator = itertools.cycle(validation_set)

    def get_next_index(validation):
        if validation:
            return next(validation_iterator)
        else:
            return next(training_iterator)

    def get_base_path(validation):
        if validation:
            return VALIDATION_PATH
        else:
            return TRAINING_PATH

    def get_input_list(validation):
        if validation:
            return ("BF", "PC")
        else:
            return ("BF", "PC", "mask")

    root = dt.DummyFeature(
        # On each update, root will grab the next value from this iterator
        index=get_next_index,
        base_path=get_base_path,
        input_list=get_input_list,
    )

    brightfield_phase_contrast_loader = dt.LoadImage(
        **root.properties,
        file_names=lambda index, input_list: [_file_name_struct.format(
            _type, index) for _type in input_list],
        path=lambda base_path, file_names: [os.path.join(
            base_path, file_name) for file_name in file_names],
    )

    fluorescence_loader = dt.LoadImage(
        **root.properties,
        file_names=lambda index: [_file_name_struct.format(
            _type, index) for _type in ("NC", "LD")],
        path=lambda base_path, file_names: [os.path.join(
            base_path, file_name) for file_name in file_names],
    )

    flip = dt.FlipLR()

    affine = dt.Affine(
        rotate=lambda: np.random.rand() * 2 * np.pi,
    )

    corner = int(512 * (np.sqrt(2) - 1) / 2)
    cropping = dt.Crop(
        crop=(512, 512, None),
        corner=(corner, corner, 0)
    )

    data_pair = dt.Combine(
        [brightfield_phase_contrast_loader, fluorescence_loader])

    padded_crop_size = int(512 * np.sqrt(2))

    def get_points(validation):
        if validation:
            return (0, 0)
        else:
            image = brightfield_phase_contrast_loader.resolve()
            _points = np.where(image[:, :, 2] != 0)
            idx = np.random.randint(len(_points[0]))

            return (_points[0][idx], _points[1][idx])

    _data_pair = data_pair + \
        dt.Merge(lambda: lambda image: [image[0][:, :, 0:2], image[1]])

    params = apido.get_dataset_parameters(
        DATASET_PATH,
        data_feature=_data_pair,
        n_images=len(training_set)
    )

    PointsDummy = dt.DummyFeature(
        corners=get_points,
    )

    cropped_data = dt.Crop(
        _data_pair,
        crop=(padded_crop_size, padded_crop_size, None),
        updates_per_reload=16,
        corner=lambda corners: (*corners, 0),
        **PointsDummy.properties
    )

    augmented_data = cropped_data + flip + affine + cropping

    validation_data = _data_pair + dt.PadToMultiplesOf(multiple=(32, 32, None))

    return dt.ConditionalSetFeature(
        on_true=validation_data,
        on_false=augmented_data,
        condition="is_validation",
        is_validation=lambda validation: validation
    ) + dt.AsType("float64")


def DataGenerator(
    min_data_size=1024,
    max_data_size=1025,
    **kwargs
):

    feature = DataLoader(**kwargs)

    conf["feature"] = feature

    args = {
        "feature": feature,
        "batch_function": lambda image: image[0],
        "label_function": lambda image: image[1],
        "min_data_size": min_data_size,
        "max_data_size": max_data_size,
        **kwargs,
    }
    return dt.utils.safe_call(dt.generators.ContinuousGenerator, **args)


def get_validation_set(validation_set_size=2):

    dataset = conf["feature"]

    validation_inputs = []
    validation_targets = []

    for _ in range(validation_set_size):
        data_tuple = dataset.update(validation=True).resolve()
        validation_inputs.append(data_tuple[0])
        validation_targets.append(data_tuple[1])

    return np.array(validation_inputs), np.array(validation_targets)
