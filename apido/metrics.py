# All metrics

from tensorflow.keras import backend as K


def mae(P, T):
    return K.mean(K.abs(P - T))


def w_nuclei(index, weight=1):
    def nuclei(P, T):
        return mae(P[..., index], T[..., index]) * weight

    return nuclei


def w_lipids(index, weight=1):
    def lipids(P, T):
        return mae(P[..., index], T[..., index]) * weight

    return lipids


_metrics = [w_nuclei, w_lipids]


def metrics():
    return [_metrics[index](index) for index in range(2)]
