# -*- coding: utf-8 -*-
# ./hsi_plots.py
"""
Created on Fri Aug  7 15:23:22 2026

@author: Lenovo
"""
from __future__ import annotations
import matplotlib.pyplot as plt
import numpy as np


# %%

def plot_all_spectra_by_classes(
    spectral_data,
    classes,
    wavelength,
    max_spectra_per_class=100,
    random_state=42,
    linewidth=0.8,
    alpha=0.5,
):

    rng = np.random.default_rng(random_state)

    unique_classes = np.unique(classes)

    # Generate one color per class
    cmap = plt.get_cmap("tab10")

    class_colors = {
        cls: cmap(i % cmap.N)
        for i, cls in enumerate(unique_classes)
    }

    plt.figure(figsize=(6, 4))

    for cls in unique_classes:

        indices = np.where(classes == cls)[0]

        if len(indices) > max_spectra_per_class:
            indices = rng.choice(
                indices,
                size=max_spectra_per_class,
                replace=False,
            )

        color = class_colors[cls]

        # plot spectra
        for k, idx in enumerate(indices):

            plt.plot(
                wavelength,
                spectral_data[idx],
                color=color,
                linewidth=linewidth,
                alpha=alpha,
                label=cls if k == 0 else None,
            )

    plt.xlabel("Wavelength (nm)", fontsize=12)
    plt.ylabel("Reflectance", fontsize=12)

    plt.title(
        "Randomly Sampled Spectra by Class",
        fontsize=14,
    )

    plt.grid(True, alpha=0.3)

    plt.legend(
        title="Classes",
        frameon=True,
    )

    plt.tight_layout()

    plt.show()


# %%

def plot_mean_spectra_by_classes(
    spectral_data,
    classes,
    wavelength,
    max_spectra_per_class=100,
    random_state=42,
    linewidth=2.0,
    band_alpha=0.20,
):

    spectral_data = np.asarray(spectral_data)
    classes = np.asarray(classes)
    wavelength = np.asarray(wavelength)

    # --------------------------------------------------------------
    # Input validation
    # --------------------------------------------------------------
    if spectral_data.ndim != 2:
        raise ValueError(
            "spectral_data must be a two-dimensional array with shape "
            "(number_of_spectra, number_of_bands)."
        )

    if classes.ndim != 1:
        raise ValueError(
            "classes must be a one-dimensional array."
        )

    if wavelength.ndim != 1:
        raise ValueError(
            "wavelength must be a one-dimensional array."
        )

    if spectral_data.shape[0] != classes.size:
        raise ValueError(
            "The number of rows in spectral_data must equal the number "
            "of elements in classes."
        )

    if spectral_data.shape[1] != wavelength.size:
        raise ValueError(
            "The number of spectral bands must equal the number of "
            "wavelength values."
        )

    if max_spectra_per_class is not None:
        if not isinstance(max_spectra_per_class, int):
            raise TypeError(
                "max_spectra_per_class must be an integer or None."
            )

        if max_spectra_per_class <= 0:
            raise ValueError(
                "max_spectra_per_class must be greater than zero."
            )

    rng = np.random.default_rng(random_state)
    unique_classes = np.unique(classes)

    # One different color per class.
    color_map = plt.get_cmap("tab10")
    class_colors = {
        class_name: color_map(index % color_map.N)
        for index, class_name in enumerate(unique_classes)
    }

    fig, ax = plt.subplots(
        figsize=(10, 6)
    )

    # --------------------------------------------------------------
    # Calculate and plot mean ± standard deviation for each class
    # --------------------------------------------------------------
    for class_name in unique_classes:
        class_indices = np.flatnonzero(
            classes == class_name
        )

        if (
            max_spectra_per_class is not None
            and class_indices.size > max_spectra_per_class
        ):
            selected_indices = rng.choice(
                class_indices,
                size=max_spectra_per_class,
                replace=False,
            )
        else:
            selected_indices = class_indices

        selected_spectra = spectral_data[
            selected_indices,
            :,
        ]

        mean_spectrum = np.nanmean(
            selected_spectra,
            axis=0,
        )

        # ddof=1 gives the sample standard deviation when at least
        # two spectra are available.
        if selected_spectra.shape[0] > 1:
            std_spectrum = np.nanstd(
                selected_spectra,
                axis=0,
                ddof=1,
            )
        else:
            std_spectrum = np.zeros_like(
                mean_spectrum
            )

        lower_band = mean_spectrum - std_spectrum
        upper_band = mean_spectrum + std_spectrum

        color = class_colors[class_name]

        ax.plot(
            wavelength,
            mean_spectrum,
            color=color,
            linewidth=linewidth,
            label=(
                f"{class_name} "
                f"(n={selected_spectra.shape[0]:,})"
            ),
        )

        ax.fill_between(
            wavelength,
            lower_band,
            upper_band,
            color=color,
            alpha=band_alpha,
            linewidth=0,
        )

    ax.set_xlabel(
        "Wavelength (nm)",
        fontsize=12,
    )

    ax.set_ylabel(
        "Reflectance",
        fontsize=12,
    )

    ax.set_title(
        "Mean Spectra by Class (Mean ± 1 SD)",
        fontsize=14,
    )

    ax.grid(
        True,
        alpha=0.3,
    )

    ax.legend(
        title="Class",
        frameon=True,
    )

    fig.tight_layout()
    plt.show()

    return fig, ax
