import os
import apido
import apido.deeptrack as dt
import numpy as np
import scipy.optimize as optimize


class Parameters(dict):
    def add(self, key, val):
        t_key = "__" + key
        if t_key not in self:
            self[t_key] = []

        self[t_key].append(val)
        self[key] = tuple(np.mean(self[t_key], axis=0))

    def bin(self, key, propname, reducer=None):
        result_dict = {}
        for value, props in zip(self["__" + key], self["props"]):

            propvalue = "NaN"
            for p in props:
                if propname in p:
                    propvalue = p[propname]
                    break

            key = str(propvalue)
            if key not in result_dict:
                result_dict[key] = []

            result_dict[key].append(value)

        if reducer:
            for key, value in result_dict.items():
                result_dict[key] = tuple(reducer(value, axis=0))
        return result_dict


def get_dataset_parameters(
    dataset_path,
    data_feature=None,
    n_images=10,
    no_load=False,
):
    config_path = os.path.join(dataset_path, "params.json")
    try:
        config = apido.load_config(config_path)
    except Exception as e:
        print(e)
        config = False

    if no_load or not config:
        config = dict(calculate_dataset_parameters(data_feature, n_images))
        apido.save_config(config_path, config)

    return Parameters(config)


def is_jsonable(x):
    import json

    try:
        json.dumps(x)
        return True
    except (TypeError, OverflowError):
        return False


def reduce_property(prop):

    if isinstance(prop, dict):
        reduced_property = {}
        for key, val in prop.items():
            setter = val
            if isinstance(val, dict):
                setter = reduce_property(val)
            if isinstance(val, list):
                setter = [reduce_property(v) for v in val]
            if isinstance(val, np.ndarray):
                setter = tuple(val)

            if is_jsonable(setter):
                reduced_property[key] = setter

        return reduced_property

    if is_jsonable(prop):
        return prop
    else:
        return None


def calculate_dataset_parameters(data_feature, n_images):

    parameters = Parameters()

    parameters["props"] = []

    for i in range(n_images):
        print("Evaluating image {0}...".format(i), end="\r")
        data = data_feature.update().resolve()

        reduced_properties = [reduce_property(
            prop) for prop in data[0].properties]

        parameters["props"].append(reduced_properties)

        parameters.add("input_subtract", get_input_subtract(*data))
        parameters.add("target_subtract", get_target_subtract(*data))
        parameters.add("input_divide", get_input_divide(*data))
        parameters.add("target_divide", get_target_divide(*data))

    return parameters


def get_input_subtract(input, target):
    return tuple(np.quantile(input, 0.01, axis=(0, 1), keepdims=False))


def get_target_subtract(input, target):
    return tuple(np.quantile(target, 0.01, axis=(0, 1), keepdims=False))


def get_input_divide(input, target):
    return tuple(np.quantile(input, 0.99, axis=(0, 1), keepdims=False))


def get_target_divide(input, target):
    return tuple(np.quantile(target, 0.99, axis=(0, 1), keepdims=False))
