import os
import glob
import shutil
import time

import numpy as np

_file_name_struct = "HepaRG_{0}_S{1}.tif"
_data_types = ["BF", "PC", "NC", "LD"]


def save_data(dataset, cntr, folder):

    filenames = glob.glob(
        os.path.join(
            ".", "datasets",
            dataset,
            "scr " + cntr,
            "*ORG.tif"
        )
    )

    current_data = int(
        len(glob.glob(os.path.join(
            ".", "datasets",
            dataset,
            folder,
            "*tif")
        )
        )/4
    )

    print(os.path.join(
        ".", "datasets",
        dataset,
        folder,
        "*tif"))

    j, c = current_data, 0
    for idx, file in enumerate(filenames):

        if idx != 0 and idx % 4 == 0:
            print("saving set: " + str(j))
            j, c = j + 1, 0

        shutil.copy(file, os.path.join(
            filenames[0][:17],
            folder,
            _file_name_struct.format(_data_types[c], ("0"+str(j))[-2:])
        ))
        c += 1

    print("saving set: " + str(j))


def split_validation(dataset, _from, _to, percentage):

    filenames = glob.glob(
        os.path.join(
            ".", "datasets",
            dataset,
            _from,
            "*.tif"
        )
    )

    current_data = int(len(filenames)/4)

    split = np.round(current_data*percentage).astype(np.int64)
    rand_idxs = [("0" + str(idx))[-2:]
                 for idx in list(np.random.choice(range(current_data), split, replace=False))]

    for idx in rand_idxs:
        validation_set = [
            file for file in filenames if file.endswith(idx + ".tif")]

        for file in validation_set:
            shutil.copy(file, file.replace(_from, _to))
            os.remove(file)


save_data(
    dataset="HepaRG", cntr="200", folder="training_data"
)

time.sleep(0.5)

save_data(
    dataset="HepaRG", cntr="400", folder="training_data"
)

time.sleep(0.5)

split_validation(
    dataset="HepaRG", _from="training_data", _to="validation_data", percentage=0.2
)
