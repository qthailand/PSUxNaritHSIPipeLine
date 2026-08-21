# -*- coding: utf-8 -*-
# ./hsi_utilities.py
"""
Created on Fri Aug  7 15:32:48 2026

@author: Lenovo
"""
import numpy as np

# %%


def get_spectra_per_class(classes):
    unique_classes, class_counts = np.unique(
        classes,
        return_counts=True,
    )
    print("\nNumber of spectra per class")
    print("---------------------------")

    for cls, count in zip(unique_classes, class_counts):
        print(f"{cls:20s}: {count:,}")

    # spectra_per_class = dict(
    #     zip(unique_classes, class_counts)
    # )

    return unique_classes, class_counts

# %% Statistical Outliner Removal (SOR)


def spectral_sor(
    spectral_data,
    classes,
    std_weight=3.0,
):
    """
    Statistical Outlier Removal (SOR) for spectra.

    Outlier detection is performed separately within each class.

    For each class:
        1. Calculate the mean spectrum.
        2. Calculate the RMS spectral distance of each spectrum
           from the class mean spectrum.
        3. Calculate the outlier threshold:

           threshold = mean_distance + std_weight * std_distance

        4. Remove spectra with distance greater than the threshold.

    Parameters
    ----------
    spectral_data : array-like, shape (n_spectra, n_bands)
        Spectral data. Each row represents one spectrum.

    classes : array-like, shape (n_spectra,)
        Class label corresponding to each spectrum.

    std_weight : float, default=3.0
        Weight applied to the standard deviation when calculating
        the outlier threshold.

        Smaller values remove more spectra.

        Examples
        --------
        3.0 : conservative
        2.0 : moderate
        1.5 : aggressive

    Returns
    -------
    filtered_spectral_data : np.ndarray
        Spectra remaining after outlier removal.

    filtered_classes : np.ndarray
        Class labels corresponding to the remaining spectra.

    keep_mask : np.ndarray of bool
        Boolean mask indicating retained spectra.

        The same mask can be applied to other sample-level
        metadata such as:

        cube_names
        image_id
        annotation_id
        sample_id
    """

    spectral_data = np.asarray(
        spectral_data
    )

    classes = np.asarray(
        classes
    )

    # --------------------------------------------------------------
    # Input validation
    # --------------------------------------------------------------

    if spectral_data.ndim != 2:

        raise ValueError(
            "spectral_data must be a 2D array with shape "
            "(n_spectra, n_bands)."
        )

    if classes.ndim != 1:

        raise ValueError(
            "classes must be a 1D array."
        )

    if spectral_data.shape[0] != classes.shape[0]:

        raise ValueError(
            "spectral_data and classes must contain the same "
            "number of spectra."
        )

    if not isinstance(
        std_weight,
        (int, float),
    ):

        raise TypeError(
            "std_weight must be a number."
        )

    if std_weight <= 0:

        raise ValueError(
            "std_weight must be greater than zero."
        )

    # --------------------------------------------------------------
    # Initialize mask
    # --------------------------------------------------------------

    keep_mask = np.ones(
        spectral_data.shape[0],
        dtype=bool,
    )

    unique_classes = np.unique(
        classes
    )

    print(
        "\nStatistical outlier removal by class"
    )

    print(
        "------------------------------------"
    )

    print(
        f"Standard deviation weight = {std_weight}"
    )

    # --------------------------------------------------------------
    # Process each class separately
    # --------------------------------------------------------------

    for cls in unique_classes:

        class_indices = np.flatnonzero(
            classes == cls
        )

        class_spectra = spectral_data[
            class_indices,
            :,
        ]

        number_of_spectra = (
            class_spectra.shape[0]
        )

        # ----------------------------------------------------------
        # Too few spectra for meaningful statistics
        # ----------------------------------------------------------

        if number_of_spectra < 3:

            print(
                f"{str(cls):20s}: "
                f"n={number_of_spectra:6,d} | "
                f"removed={0:6,d} | "
                f"retained={number_of_spectra:6,d}"
            )

            continue

        # ----------------------------------------------------------
        # Mean spectrum of the current class
        # ----------------------------------------------------------

        class_mean_spectrum = np.nanmean(
            class_spectra,
            axis=0,
        )

        # ----------------------------------------------------------
        # RMS spectral distance
        #
        # Each spectrum gets one distance value describing how far
        # its entire spectral curve is from the class mean.
        # ----------------------------------------------------------

        spectral_difference = (
            class_spectra
            - class_mean_spectrum
        )

        distances = np.sqrt(
            np.nanmean(
                spectral_difference ** 2,
                axis=1,
            )
        )

        # ----------------------------------------------------------
        # Distance statistics
        # ----------------------------------------------------------

        mean_distance = np.nanmean(
            distances
        )

        std_distance = np.nanstd(
            distances,
            ddof=1,
        )

        # ----------------------------------------------------------
        # SOR threshold
        # ----------------------------------------------------------

        threshold = (
            mean_distance
            + std_weight * std_distance
        )

        # ----------------------------------------------------------
        # Identify spectra to retain
        # ----------------------------------------------------------

        class_keep_mask = (
            np.isfinite(distances)
            & (distances <= threshold)
        )

        keep_mask[
            class_indices
        ] = class_keep_mask

        # ----------------------------------------------------------
        # Statistics
        # ----------------------------------------------------------

        removed_count = int(
            np.sum(
                ~class_keep_mask
            )
        )

        retained_count = int(
            np.sum(
                class_keep_mask
            )
        )

        print(
            f"{str(cls):20s}: "
            f"n={number_of_spectra:6,d} | "
            f"removed={removed_count:6,d} | "
            f"retained={retained_count:6,d}"
        )

    # --------------------------------------------------------------
    # Apply final mask
    # --------------------------------------------------------------

    filtered_spectral_data = spectral_data[
        keep_mask,
        :,
    ]

    filtered_classes = classes[
        keep_mask
    ]

    # --------------------------------------------------------------
    # Summary
    # --------------------------------------------------------------

    total_removed = int(
        np.sum(
            ~keep_mask
        )
    )

    print(
        "------------------------------------"
    )

    print(
        f"Total spectra before : "
        f"{spectral_data.shape[0]:,}"
    )

    print(
        f"Total spectra removed: "
        f"{total_removed:,}"
    )

    print(
        f"Total spectra after  : "
        f"{filtered_spectral_data.shape[0]:,}"
    )

    return (
        filtered_spectral_data,
        filtered_classes,
        keep_mask,
    )

# %% Savitzky–Golay Smoothening


def spectral_SG_smooth(
    spectral_data,
    classes,
    window_length=11,
    polyorder=2,
):
    """
    Smooth hyperspectral spectra using the Savitzky-Golay filter.

    The filtering is performed using vectorized processing along
    the spectral-band axis. No loop over individual spectra is used.

    Parameters
    ----------
    spectral_data : array-like, shape (n_spectra, n_bands)
        Spectral data. Each row represents one spectrum.

    classes : array-like, shape (n_spectra,)
        Class label corresponding to each spectrum.
        Class labels are not modified by the smoothing process.

    window_length : int, default=11
        Number of neighboring spectral bands used for Savitzky-Golay
        smoothing.

        The value must be:
            - odd
            - greater than polyorder
            - smaller than or equal to the number of spectral bands

        Larger values produce stronger smoothing.

        Typical values:
            5   = very light smoothing
            7   = light smoothing
            11  = moderate smoothing
            15  = stronger smoothing
            21  = very strong smoothing

    polyorder : int, default=2
        Polynomial order used to locally fit the spectra.

        Typical values:
            2 = recommended default
            3 = preserve more complex spectral shapes

    Returns
    -------
    spectral_data_smooth : np.ndarray
        Smoothed spectral data with the same shape as spectral_data.

    classes_smooth : np.ndarray
        Class labels corresponding to the smoothed spectra.
        These are unchanged from the input classes.
    """

    import numpy as np
    from scipy.signal import savgol_filter

    # --------------------------------------------------------------
    # Convert inputs to NumPy arrays
    # --------------------------------------------------------------

    spectral_data = np.asarray(
        spectral_data
    )

    classes = np.asarray(
        classes
    )

    # --------------------------------------------------------------
    # Input validation
    # --------------------------------------------------------------

    if spectral_data.ndim != 2:

        raise ValueError(
            "spectral_data must be a 2D array with shape "
            "(n_spectra, n_bands)."
        )

    if classes.ndim != 1:

        raise ValueError(
            "classes must be a 1D array."
        )

    if spectral_data.shape[0] != classes.shape[0]:

        raise ValueError(
            "spectral_data and classes must contain "
            "the same number of spectra."
        )

    if not isinstance(
        window_length,
        int,
    ):

        raise TypeError(
            "window_length must be an integer."
        )

    if window_length < 3:

        raise ValueError(
            "window_length must be at least 3."
        )

    if window_length % 2 == 0:

        raise ValueError(
            "window_length must be an odd integer."
        )

    if window_length > spectral_data.shape[1]:

        raise ValueError(
            "window_length cannot be greater than "
            "the number of spectral bands."
        )

    if not isinstance(
        polyorder,
        int,
    ):

        raise TypeError(
            "polyorder must be an integer."
        )

    if polyorder < 0:

        raise ValueError(
            "polyorder must be >= 0."
        )

    if polyorder >= window_length:

        raise ValueError(
            "polyorder must be smaller than window_length."
        )

    # --------------------------------------------------------------
    # Vectorized Savitzky-Golay filtering
    #
    # spectral_data shape:
    #
    #     (n_spectra, n_bands)
    #
    # axis=1 means the filter moves along wavelength/band direction.
    #
    # All spectra are processed simultaneously.
    # --------------------------------------------------------------

    spectral_data_smooth = savgol_filter(
        spectral_data,
        window_length=window_length,
        polyorder=polyorder,
        axis=1,
        mode="interp",
    )

    # Preserve compact floating-point representation
    spectral_data_smooth = spectral_data_smooth.astype(
        spectral_data.dtype,
        copy=False,
    )

    # --------------------------------------------------------------
    # Information
    # --------------------------------------------------------------

    print(
        "\nSavitzky-Golay spectral smoothing"
    )

    print(
        "---------------------------------"
    )

    print(
        f"Number of spectra : "
        f"{spectral_data.shape[0]:,}"
    )

    print(
        f"Number of bands   : "
        f"{spectral_data.shape[1]:,}"
    )

    print(
        f"Window length     : "
        f"{window_length}"
    )

    print(
        f"Polynomial order  : "
        f"{polyorder}"
    )

    # --------------------------------------------------------------
    # Return
    # --------------------------------------------------------------

    return (
        spectral_data_smooth,
        classes,
    )


# %% Savitzky-Golay 1st Derivative
def spectral_SG_1stDeriv(
    spectral_data,
    classes,
    wavelength,
    window_length=11,
    polyorder=2,
    max_spacing_variation=5.0,
):
    """
    Calculate the first-order spectral derivative using a
    Savitzky-Golay filter.

    Parameters
    ----------
    spectral_data : array-like, shape (n_spectra, n_bands)
        Spectral data. Each row is one spectrum.

    classes : array-like, shape (n_spectra,)
        Class label corresponding to each spectrum.

    wavelength : array-like, shape (n_bands,)
        Wavelength values corresponding to spectral bands.
        Values must be finite and strictly increasing.

    window_length : int, default=11
        Savitzky-Golay window length.
        Must be odd, >= 3, > polyorder, and <= n_bands.

    polyorder : int, default=2
        Polynomial order used for local fitting.
        Must be >= 1 and < window_length.

    max_spacing_variation : float, default=5.0
        Maximum allowed coefficient of variation (%) of wavelength
        spacing. Savitzky-Golay assumes approximately uniform spacing.

    Returns
    -------
    spectral_data_1stDeriv : np.ndarray
        First derivative spectra with shape
        (n_spectra, n_bands).

    classes_out : np.ndarray
        Copy of class labels.

    Notes
    -----
    The derivative is approximately dR/d(lambda).

    For slightly non-uniform wavelength sampling, the median wavelength
    spacing is used as the Savitzky-Golay delta.
    """

    import numpy as np
    from scipy.signal import savgol_filter

    # ------------------------------------------------------------
    # 1. Convert input
    # ------------------------------------------------------------
    spectral_data = np.asarray(spectral_data)
    classes = np.asarray(classes)
    wavelength = np.asarray(wavelength, dtype=np.float64)

    # ------------------------------------------------------------
    # 2. Basic dimension validation
    # ------------------------------------------------------------
    if spectral_data.ndim != 2:
        raise ValueError(
            "spectral_data must be a 2D array with shape "
            "(n_spectra, n_bands). "
            f"Received shape {spectral_data.shape}."
        )

    if classes.ndim != 1:
        raise ValueError(
            "classes must be a 1D array. "
            f"Received shape {classes.shape}."
        )

    if wavelength.ndim != 1:
        raise ValueError(
            "wavelength must be a 1D array. "
            f"Received shape {wavelength.shape}."
        )

    n_spectra, n_bands = spectral_data.shape

    if n_spectra == 0:
        raise ValueError(
            "spectral_data contains no spectra."
        )

    if n_bands < 3:
        raise ValueError(
            "At least 3 spectral bands are required."
        )

    # ------------------------------------------------------------
    # 3. Shape consistency
    # ------------------------------------------------------------
    if classes.size != n_spectra:
        raise ValueError(
            "Number of class labels does not match number of spectra: "
            f"{classes.size} labels vs {n_spectra} spectra."
        )

    if wavelength.size != n_bands:
        raise ValueError(
            "Number of wavelength values does not match number of bands: "
            f"{wavelength.size} wavelengths vs {n_bands} bands."
        )

    # ------------------------------------------------------------
    # 4. Numeric data validation
    # ------------------------------------------------------------
    if not np.issubdtype(spectral_data.dtype, np.number):
        raise TypeError(
            "spectral_data must contain numeric values."
        )

    if not np.all(np.isfinite(spectral_data)):
        n_invalid = np.size(spectral_data) - np.count_nonzero(
            np.isfinite(spectral_data)
        )

        raise ValueError(
            "spectral_data contains NaN or infinite values "
            f"({n_invalid:,} invalid values). "
            "Clean or remove invalid spectra before calculating "
            "the derivative."
        )

    if not np.all(np.isfinite(wavelength)):
        raise ValueError(
            "wavelength contains NaN or infinite values."
        )

    # ------------------------------------------------------------
    # 5. Validate wavelength axis
    # ------------------------------------------------------------
    wavelength_spacing = np.diff(wavelength)

    if np.any(wavelength_spacing <= 0):
        raise ValueError(
            "wavelength values must be strictly increasing "
            "with no duplicated bands."
        )

    mean_spacing = float(
        np.mean(wavelength_spacing)
    )

    median_spacing = float(
        np.median(wavelength_spacing)
    )

    std_spacing = float(
        np.std(wavelength_spacing)
    )

    if mean_spacing <= 0:
        raise ValueError(
            "Invalid wavelength spacing."
        )

    spacing_variation = (
        std_spacing / mean_spacing * 100.0
    )

    if not isinstance(
        max_spacing_variation,
        (int, float, np.integer, np.floating),
    ):
        raise TypeError(
            "max_spacing_variation must be numeric."
        )

    if max_spacing_variation < 0:
        raise ValueError(
            "max_spacing_variation must be >= 0."
        )

    if spacing_variation > max_spacing_variation:
        raise ValueError(
            "Wavelength spacing is too non-uniform for reliable "
            "Savitzky-Golay differentiation.\n"
            f"Spacing variation = {spacing_variation:.2f}%\n"
            f"Allowed maximum   = {max_spacing_variation:.2f}%\n"
            "Consider resampling the spectra onto a uniformly spaced "
            "wavelength grid first."
        )

    # ------------------------------------------------------------
    # 6. Validate window_length
    # ------------------------------------------------------------
    if not isinstance(
        window_length,
        (int, np.integer),
    ):
        raise TypeError(
            "window_length must be an integer."
        )

    window_length = int(window_length)

    if window_length < 3:
        raise ValueError(
            "window_length must be >= 3."
        )

    if window_length % 2 == 0:
        raise ValueError(
            "window_length must be an odd integer."
        )

    if window_length > n_bands:
        raise ValueError(
            f"window_length ({window_length}) cannot exceed "
            f"the number of spectral bands ({n_bands})."
        )

    # ------------------------------------------------------------
    # 7. Validate polyorder
    # ------------------------------------------------------------
    if not isinstance(
        polyorder,
        (int, np.integer),
    ):
        raise TypeError(
            "polyorder must be an integer."
        )

    polyorder = int(polyorder)

    if polyorder < 1:
        raise ValueError(
            "polyorder must be >= 1 for first derivative."
        )

    if polyorder >= window_length:
        raise ValueError(
            "polyorder must be smaller than window_length."
        )

    # ------------------------------------------------------------
    # 8. Convert data to safe floating-point representation
    # ------------------------------------------------------------
    if spectral_data.dtype == np.float64:
        working_data = spectral_data
        output_dtype = np.float64

    else:
        working_data = spectral_data.astype(
            np.float32,
            copy=False,
        )
        output_dtype = np.float32

    # ------------------------------------------------------------
    # 9. Savitzky-Golay first derivative
    # ------------------------------------------------------------
    try:
        spectral_data_1stDeriv = savgol_filter(
            working_data,
            window_length=window_length,
            polyorder=polyorder,
            deriv=1,
            delta=median_spacing,
            axis=1,
            mode="interp",
        )

    except Exception as exc:
        raise RuntimeError(
            "Savitzky-Golay first derivative calculation failed. "
            f"Original error: {exc}"
        ) from exc

    # ------------------------------------------------------------
    # 10. Final output validation
    # ------------------------------------------------------------
    spectral_data_1stDeriv = np.asarray(
        spectral_data_1stDeriv,
        dtype=output_dtype,
    )

    if spectral_data_1stDeriv.shape != spectral_data.shape:
        raise RuntimeError(
            "Unexpected output shape after Savitzky-Golay derivative: "
            f"input={spectral_data.shape}, "
            f"output={spectral_data_1stDeriv.shape}."
        )

    if not np.all(np.isfinite(spectral_data_1stDeriv)):
        raise RuntimeError(
            "Savitzky-Golay derivative produced NaN or infinite values."
        )

    # ------------------------------------------------------------
    # 11. Information
    # ------------------------------------------------------------
    print("\nSavitzky-Golay 1st derivative")
    print("--------------------------------")
    print(f"Number of spectra        : {n_spectra:,}")
    print(f"Number of bands          : {n_bands:,}")
    print(f"Window length            : {window_length}")
    print(f"Polynomial order         : {polyorder}")
    print(f"Mean wavelength spacing  : {mean_spacing:.4f}")
    print(f"Median wavelength spacing: {median_spacing:.4f}")
    print(f"Spacing variation        : {spacing_variation:.2f}%")
    print(
        "Output shape             : "
        f"{spectral_data_1stDeriv.shape}"
    )

    # ------------------------------------------------------------
    # 12. Return
    # ------------------------------------------------------------
    return (
        spectral_data_1stDeriv,
        classes,
    )
