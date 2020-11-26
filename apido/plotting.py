import matplotlib.pyplot as plt
import os
import apido
import numpy as np
import glob


def plot_evaluation(brightfield, target, prediction, ncols=5):
    plt.figure(figsize=(2 * target.shape[-1] * ncols, 2.5 * (target.shape[-1] + 1)))
    for col in range(ncols):
        plt.subplot(1 + target.shape[-1], ncols, col + 1)
        plt.imshow(brightfield[col, :, :, 3], vmin=0, vmax=4000)
        plt.axis("off")

        for row in range(target.shape[-1]):
            plt.subplot(
                1 + target.shape[-1],
                ncols * 2,
                ncols * (row + 1) * 2 + 1 + col * 2,
            )
            plt.imshow(prediction[col, :, :, row], vmin=0, vmax=4000)
            plt.axis("off")
            plt.subplot(
                1 + target.shape[-1],
                ncols * 2,
                ncols * (row + 1) * 2 + 2 + col * 2,
            )
            plt.imshow(target[col, :, :, row], vmin=0, vmax=4000)
            plt.axis("off")

        plt.subplots_adjust(hspace=0.02, wspace=0.02)
    return plt
