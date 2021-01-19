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

        parameters.add("offset", get_per_image_correction(*data))
    return parameters


def get_input_subtract(input, target):
    return tuple(np.quantile(input, 0.01, axis=(0, 1), keepdims=False))


def get_target_subtract(input, target):
    return tuple(np.quantile(target, 0.01, axis=(0, 1), keepdims=False))


def get_input_divide(input, target):
    return tuple(np.quantile(input, 0.99, axis=(0, 1), keepdims=False))


def get_target_divide(input, target):
    return tuple(np.quantile(target, 0.99, axis=(0, 1), keepdims=False))


def get_per_image_correction(input, target):
    def fit_function(X, a, b, c, d, e=0):
        x = X[0]
        y = X[1]
        return a * np.exp(-((x - b) ** 2 + (y - c) ** 2) / (2 * d ** 2)) + e

    x_corr = 0
    y_corr = 0

    step = 512
    correlation_fields = (0, 1)

    inner_size = 8

    take_center = (
        slice(step // 2 - inner_size, step // 2 + inner_size),
        slice(step // 2 - inner_size, step // 2 + inner_size),
    )

    _x = np.arange(-step / 2, step / 2)
    X, Y = np.meshgrid(_x, _x)

    Xsmall = X[take_center]
    Ysmall = Y[take_center]

    results = []
    corrected_input = dt.Affine(translate=(
        x_corr, y_corr)).resolve(np.array(input))
    for x in range(10, input.shape[0] - step - 20, step):
        for y in range(0, input.shape[1] - step, step):
            I_1 = corrected_input[x: x + step,
                                  y: y + step, correlation_fields[0]]
            I_2 = target[x: x + step, y: y + step, correlation_fields[1]]
            I_1 = np.fft.fft2(I_1)
            I_2 = np.fft.fft2(I_2)

            fft_correlation = I_1 * np.conjugate(I_2)
            correlation = np.abs(
                np.fft.fftshift(np.fft.ifft2(fft_correlation))[take_center]
            )
            max_val = np.max(correlation)
            med_val = np.mean(correlation)

            try:
                a, r = optimize.curve_fit(
                    lambda x, a, b, c, d: fit_function(x, a, b, c, d, med_val),
                    np.array([Xsmall.flatten(), Ysmall.flatten()]),
                    correlation.flatten(),
                    [max_val, 0, 0, 5],
                )

                results.append((a[1], a[2]))
            except Exception as e:
                pass

    x_err, y_err = np.median(results, axis=0)

    return -x_err, -y_err
