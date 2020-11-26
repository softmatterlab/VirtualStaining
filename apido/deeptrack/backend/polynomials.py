""" Expands the set of polynomials available through scipy
"""

import numpy as np
from scipy.special import jv, h1vp, yv


def ricbesj(n, x):
    """The Riccati-Bessel polynomial of the first kind.

    Parameters
    ----------
    n : int, float
        Polynomial order
    x : number, ndarray
        The point(s) the polynomial is evaluated at

    Returns
    -------
    float, ndarray
        The polynomial evaluated at x
    """

    return np.sqrt(np.pi * x / 2) * besselj(n + 0.5, x)


def dricbesj(n, x):
    """The first derivative of the Riccati-Bessel polynomial of the first kind.

    Parameters
    ----------
    n : int, float
        Polynomial order
    x : number, ndarray
        The point(s) the polynomial is evaluated at

    Returns
    -------
    float, ndarray
        The polynomial evaluated at x
    """

    return 0.5 * np.sqrt(np.pi / x / 2) * besselj(n + 0.5, x) + np.sqrt(
        np.pi * x / 2
    ) * dbesselj(n + 0.5, x)


def ricbesy(n, x):
    """The Riccati-Bessel polynomial of the second kind.

    Parameters
    ----------
    n : int, float
        Polynomial order
    x : number, ndarray
        The point(s) the polynomial is evaluated at

    Returns
    -------
    float, ndarray
        The polynomial evaluated at x
    """

    return -np.sqrt(np.pi * x / 2) * bessely(n + 0.5, x)


def dricbesy(n, x):
    """The first derivative of the Riccati-Bessel polynomial of the second kind.

    Parameters
    ----------
    n : int, float
        Polynomial order
    x : number, ndarray
        The point(s) the polynomial is evaluated at

    Returns
    -------
    float, ndarray
        The polynomial evaluated at x
    """

    return -0.5 * np.sqrt(np.pi / 2 / x) * yv(n + 0.5, x) - np.sqrt(
        np.pi * x / 2
    ) * dbessely(n + 0.5, x)


def ricbesh(n, x):
    """The Riccati-Bessel polynomial of the third kind.

    Parameters
    ----------
    n : int, float
        Polynomial order
    x : number, ndarray
        The point(s) the polynomial is evaluated at

    Returns
    -------
    float, ndarray
        The polynomial evaluated at x
    """
    return np.sqrt(np.pi * x / 2) * h1vp(n + 0.5, x, False)


def dricbesh(n, x):
    """The first derivative of the Riccati-Bessel polynomial of the third kind.

    Parameters
    ----------
    n : int, float
        Polynomial order
    x : number, ndarray
        The point(s) the polynomial is evaluated at

    Returns
    -------
    float, ndarray
        The polynomial evaluated at x
    """
    xi = 0.5 * np.sqrt(np.pi / 2 / x) * h1vp(n + 0.5, x, False) + np.sqrt(
        np.pi * x / 2
    ) * h1vp(n + 0.5, x, True)
    return xi


def besselj(n, x):
    """The Bessel polynomial of the first kind.

    Parameters
    ----------
    n : int, float
        Polynomial order
    x : number, ndarray
        The point(s) the polynomial is evaluated at

    Returns
    -------
    float, ndarray
        The polynomial evaluated at x
    """

    return jv(n, x)


def dbesselj(n, x):
    """The first derivative of the Bessel polynomial of the first kind.

    Parameters
    ----------
    n : int, float
        Polynomial order
    x : number, ndarray
        The point(s) the polynomial is evaluated at

    Returns
    -------
    float, ndarray
        The polynomial evaluated at x
    """
    return 0.5 * (besselj(n - 1, x) - besselj(n + 1, x))


def bessely(n, x):
    """The Bessel polynomial of the second kind.

    Parameters
    ----------
    n : int, float
        Polynomial order
    x : number, ndarray
        The point(s) the polynomial is evaluated at

    Returns
    -------
    float, ndarray
        The polynomial evaluated at x
    """

    return yv(n, x)


def dbessely(n, x):
    """The first derivative of the Bessel polynomial of the second kind.

    Parameters
    ----------
    n : int, float
        Polynomial order
    x : number, ndarray
        The point(s) the polynomial is evaluated at

    Returns
    -------
    float, ndarray
        The polynomial evaluated at x
    """
    return 0.5 * (bessely(n - 1, x) - bessely(n + 1, x))
