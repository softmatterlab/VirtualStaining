import apido
import itertools
import os

from tensorflow.keras.optimizers import Adam
from apido import deeptrack as dt

TEST_VARIABLES = {
    "generator_depth": [6],
    "generator_breadth": [16],
    "discriminator_depth": [5],
    "mae_loss_weight": [1],
    "magnification": ["40x"],
    "batch_size": [8],
    "min_data_size": [64],
    "max_data_size": [65],
    "path": [r"C:/GU/VirtualStaining/datasets"],
}


def model_initializer(
    generator_depth,
    generator_breadth,
    discriminator_depth,
    mae_loss_weight=1,
    path=None,
    magnification="60x",
    **kwargs,
):
    DATASET_PATH = os.path.join(path, "apidocytes_" + magnification)
    params = apido.get_dataset_parameters(DATASET_PATH)

    generator = apido.generator(generator_breadth, generator_depth, params)
    discriminator = apido.discriminator(discriminator_depth, params)

    return dt.models.cgan(
        generator=generator,
        discriminator=discriminator,
        discriminator_loss="mse",
        discriminator_optimizer=Adam(lr=0.0002, beta_1=0.5),
        assemble_loss=["mse", "mae"],
        assemble_optimizer=Adam(lr=0.0002, beta_1=0.5),
        assemble_loss_weights=[
            1 - mae_loss_weight,
            mae_loss_weight
        ],
    )


# Populate models
_models = []
_generators = []


def append_model(**arguments):
    _models.append((arguments, lambda: model_initializer(**arguments)))


def append_generator(**arguments):

    _generators.append(
        (
            arguments,
            lambda: apido.DataGenerator(
                ** arguments,
            ),
        )
    )


for prod in itertools.product(*TEST_VARIABLES.values()):

    arguments = dict(zip(TEST_VARIABLES.keys(), prod))
    append_model(**arguments)
    append_generator(**arguments)


def get_model(i):
    try:
        i = int(i)
    except ValueError:
        pass

    args, model = _models[i]
    return args, model()


def get_generator(i):
    try:
        i = int(i)
    except ValueError:
        pass

    args, generator = _generators[i]
    return args, generator()