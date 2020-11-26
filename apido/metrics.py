# All metrics

from tensorflow.keras import backend as K


def mae(P, T):
    return K.mean(K.abs(P - T))


norm_factor = {
    "20x": [1 / 265, 1 / 594, 1 / 641],
    "40x": [1 / 232, 1 / 615, 1 / 369],
    "60x": [1 / 201, 1 / 1201, 1 / 312],
}


def w_nuclei(weight, index):
    def nuclei(P, T):
        return mae(P[..., index], T[..., index]) * weight

    return nuclei


def w_lipids(weight, index):
    def lipids(P, T):
        return mae(P[..., index], T[..., index]) * weight

    return lipids


def w_cyto(weight, index):
    def cyto(P, T):
        return mae(P[..., index], T[..., index]) * weight

    return cyto


_metrics = [w_nuclei, w_lipids, w_cyto]


def metrics(weights="20x", actions=[1, 2, 3]):
    weights = [norm_factor[weights][action - 1] for action in actions]
    return [
        _metrics[action - 1](weight, index)
        for index, weight, action in zip(range(10), weights, actions)
    ]


def combined_metric(weights="20x", actions=[1, 2, 3]):
    """Weighted sum of metrics

    Parameters
    ----------
    weights : list, optional
        A list of weights for the individual metrics. Should be the same
        length as returned by `metrics`.

    Returns
    -------
    Callable[Tensor, Tensor] -> Tensor
        A tensorflow/keras metrics function.
    """

    def inner(P, T):

        return sum(metric(P, T) for metric in metrics(weights, actions))

    return inner
