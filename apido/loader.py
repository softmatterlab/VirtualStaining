import os
import re
import glob
import random
import itertools
import apido
import numpy as np

from apido import deeptrack as dt

VALIDATION_WELLS_AND_SITES = {
    "60x": [
        ("B03", 2),
        ("C04", 4),
        ("B04", 3),
        ("C02", 1),
        ("D02", 3),
        ("B03", 1),
        ("D04", 12),
        ("B04", 1),
        ("B03", 10),
        ("B04", 4),
        ("C02", 4),
        ("D02", 9),
        ("C04", 9),
        ("D04", 1),
        ("C02", 6),
    ],
    "40x": [
        ("B04", 4),
        ("C03", 8),
        ("C02", 8),
        ("C04", 1),
        ("C02", 5),
        ("B03", 1),
        ("B03", 4),
        ("B03", 7),
        ("D04", 5),
        ("C04", 2),
    ],
    "20x": [
        ("B04", 1),
        ("D03", 5),
        ("B04", 2),
        ("C04", 2),
        ("C03", 5),
        ("D02", 5),
        ("C02", 1),
        ("C04", 3),
    ],
}

NUMBER_OF_SITES = {
    "60x": 12,
    "40x": 8,
    "20x": 6
}

_file_name_struct = "AssayPlate_Greiner_#655090_{0}_T0001F{1}L01A0{2}Z0{3}C0{2}.tif"

conf = {}


def DataLoader(
        path=None,
        magnification="60x",
        **kwargs):

    DATASET_PATH = os.path.join(path, "apidocytes_" + magnification)
    TRAINING_PATH = os.path.join(DATASET_PATH, "training_data")

    _glob_results = glob.glob(TRAINING_PATH)

    if len(_glob_results) == 0:
        raise ValueError(
            "No path found matching glob {0}".format(TRAINING_PATH))

    elif len(_glob_results) > 1:
        from warnings import warn
        warn("Multiple paths found! Using {0}".format(TRAINING_PATH[-1]))

    print("Loading images from: \t", TRAINING_PATH)

    wells_and_sites = list(
        itertools.product(
            ["B03", "B04", "C02", "C03", "C04", "D02", "D03", "D04"],
            range(1, NUMBER_OF_SITES[magnification] + 1)
        )
    )

    random.seed(1)
    random.shuffle(wells_and_sites)

    validation_set = VALIDATION_WELLS_AND_SITES[magnification]
    training_set = [
        w_s_tuple for w_s_tuple in wells_and_sites if w_s_tuple not in validation_set]

    print("Training on {0} images".format(len(training_set)))
    print("Validating on {0} images".format(len(validation_set)))

    training_iterator = itertools.cycle(training_set)
    validation_iterator = itertools.cycle(validation_set)

    def get_next_well_and_site(validation):
        if validation:
            return next(validation_iterator)
        else:
            return next(training_iterator)

    # Accepts a tuple of form (well, site), and returns the well
    def get_well_from_tuple(well_site_tuple):
        return well_site_tuple[0]

    # Accepts a tuple of form (well, site), and returns the site as
    # a string formated to be of length 3.
    def get_site_from_tuple(well_site_tuple):
        site_string = "00" + str(well_site_tuple[1])
        return site_string[-3:]

    root = dt.DummyFeature(
        # On each update, root will grab the next value from this iterator
        well_site_tuple=get_next_well_and_site,
        # Grabs the well from the well_site_tuple
        well=get_well_from_tuple,
        # Grabs and formats the site from the well_site_tuple
        site=get_site_from_tuple,
    )

    brightfield_loader = dt.LoadImage(
        **root.properties,
        file_names=lambda well, site: [_file_name_struct.format(
            well, site, 4, z) for z in range(1, 8)],
        path=lambda file_names: [os.path.join(
            TRAINING_PATH, file_name) for file_name in file_names]
    )

    fluorescence_loader = dt.LoadImage(
        **root.properties,
        file_names=lambda well, site: [_file_name_struct.format(
            well, site, action, 1) for action in range(1, 4)],
        path=lambda file_names: [os.path.join(
            TRAINING_PATH, file_name) for file_name in file_names],
    )

    data_feature = dt.Combine([brightfield_loader, fluorescence_loader])

    params = apido.get_dataset_parameters(
        DATASET_PATH,
        data_feature=data_feature,
        n_images=len(training_set)
    )

    binned_offsets = params.bin("offset", "site", reducer=np.mean)

    correct_offset = dt.Affine(
        translate=lambda site: binned_offsets[site],
        **root.properties
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

    corrected_brightfield = brightfield_loader + correct_offset

    data_pair = dt.Combine([corrected_brightfield, fluorescence_loader])

    padded_crop_size = int(512 * np.sqrt(2))

    cropped_data = dt.Crop(
        data_pair,
        crop=(padded_crop_size, padded_crop_size, None),
        updates_per_reload=16,
        corner=lambda: (*np.random.randint(5000, size=2), 0),
    )

    augmented_data = cropped_data + flip + affine + cropping

    validation_data = data_pair + dt.PadToMultiplesOf(multiple=(32, 32, None))

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
