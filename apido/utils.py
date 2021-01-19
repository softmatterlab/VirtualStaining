import glob
import json
import re
import os
import csv
import datetime
import numpy as np
import itertools

from tensorflow import keras
import tensorflow
from tensorflow.keras import layers


import apido

_checkpoint_struct = "{0}_{1}_model_{2}"
_datestring_struct = "%d-%m-%YT%H%M%S"
_datestring_struct_old = "%d-%m-%YT%H;%M;%S"


def save_history_as_csv(path: str, history: dict, delimiter="\t"):
    """Saves the result of a keras training session as a csv.

    The output format is
    .. codeblock:

       loss{delim}val_loss{delim}...
       0.1{delim}0.11{delim}...
       0.09{delim}0.1{delim}...

    Parameters
    ----------
    path : str
        Path to save the file to.
    history : dict
        A history object created by a keras training session.
    headers : dict
        Extra heading at the top of the csv
    delimiter : str
        Delimination character placed between entries. Defaults to tab.

    """

    assert path.endswith(".csv"), "File format needs to be .csv"

    num_rows = np.inf

    for v in history.values():
        num_rows = np.min([num_rows, len(v)])

    with open(path, "w", newline="") as f:
        writer = csv.writer(f, delimiter=delimiter)

        writer.writerow(history.keys())

        # Write rows
        for idx in range(int(num_rows)):
            writer.writerow([v[idx] for v in history.values()])


def read_csv(path, delimiter="\t"):
    """Read apido-style csv file.

    Parameters
    ----------
    path : str
        path to csv file
    delimiter : str
        Delimination character placed between entries. Defaults to tab.

    Returns
    -------
    dict
        A dict of lists, where each key corresponds to one column.
    """
    result_dict = {}
    with open(path, "r") as f:
        reader = csv.reader(f, delimiter=delimiter)

        keys = next(reader)
        for key in keys:
            result_dict[key] = []

        # Write rows
        for row in reader:

            try:
                [float(r) for r in row]
            except ValueError:
                keys = row
                for key in keys:
                    result_dict[key] = []
                continue
            for key, value in zip(keys, row):
                result_dict[key].append(float(value))

    return result_dict


_folder_struct = "loss={0}_{1}_model_{2}"
_model_name = "generator_checkpoint"
_lipids_name = "lipids"
_cyto_name = "cytoplasm"
_nuclei_name = "nuclei"
_csv_name = "training_history.csv"
_config_name = "config.json"
_image_name = "comparison.png"


def save_config(path, headers):
    """Saves a config file as JSON

    Parameters
    ----------
    path : str
        Path to save to
    headers : dict
        Config dict to save
    """
    with open(path, "w") as f:
        json.dump(headers, f, indent=2)


def load_config(path):
    """Loads a JSON config file

    Parameters
    ----------
    path : str
        Path to load from.
    """
    with open(path, "r") as f:
        res = json.load(f)
    return res


def get_datestring(dtime=None):
    """Return a formated string displaying the current date."""

    if dtime is None:
        dtime = datetime.datetime.now()

    return dtime.strftime(_datestring_struct)


def get_date_from_filename(filename: str):
    """Extract datetime object from filename"""
    substrings = filename.split("_")
    for substr in reversed(substrings):

        try:
            return datetime.datetime.strptime(substr, _datestring_struct)
        except ValueError:
            try:
                return datetime.datetime.strptime(substr, _datestring_struct_old)
            except ValueError:
                pass

    raise ValueError("Could not parse date from file: " + filename)


def get_checkpoint_name(index: int or str, name: str = None):
    """Return a filename corresponding to a training configuration.

    Paramterers
    -----------
    index : int or str
        Index used for
    """
    if name is None:
        name = get_user_name()

    return _checkpoint_struct.format(name, get_datestring(), index)


_RE_PATTERN = r"([0-9]*):([0-9]*)(?::([0-9]*)){0,1}"


def parse_index(index: str):
    """Converts string argument to index iterator

    Accepts python style index selection, e.g. "1:4", ":3", "::2"
    and returns an iterator over those indexes.

    Parameters
    ----------
    index : str
        Index-string to convert to iterator
    """

    try:
        return iter([int(index)])
    except ValueError:
        match = re.match(_RE_PATTERN, index)

        if match is None:
            raise ValueError(
                "Could not parse index. Valid examples include: 0, 4:8, :6, 4:, 4:16:4 etc."
            )
        start = int(match.group(1) or 0)
        stop = int(match.group(2) or -1)
        step = int(match.group(3) or 1)

        if stop == -1:
            return itertools.count(start, step)
        else:

            return range(start, stop, step)


def get_user_name():
    """
    Retrieves the name of the user. Tries `getpass` and falls back to looking
    for a user folder in the current path.
    """
    try:
        import getpass

        return getpass.getuser()

    except Exception:
        import os

        here = os.path.abspath(".").split(os.pathsep)
        for idx, dirname in enumerate(here):
            if dirname in ("user", "User", "Users", "users"):
                return here[idx + 1]
        return "unknown"


def load_model(folder):
    """Loads a model from path.

    The path needs to contain at least a folder named `generator_checkpoint`.
    It may optionaly also include floders `lipids`, `nuclei` or `cytoplasm`,
    each of which contain a folder `generator_checkpoint`. If this is the case,
    these models will serve as stage 2 models.

    Parameters
    ----------
    folder : str
        Path to model

    """
    import warnings

    warnings.filterwarnings("ignore")

    folder = os.path.normpath(folder)

    assert os.path.exists(folder), "Folder {0} does not exist".format(folder)

    model_path = os.path.join(folder, _model_name)
    assert os.path.exists(model_path), "Folder {0} does not contain a model".format(
        folder
    )

    model = keras.models.load_model(model_path, compile=False)

    warnings.filterwarnings("default")
    return model
