# -*- coding: utf-8 -*-

adasddeadasdassdddd ตุ๊ต๊ะ
"""
T_TT_T สวัสดีครับท่านสมาชิก
Memory-efficient BIP hyperspectral spectra loader using COCO ground truth.

This version limits the number of spectra per COCO annotation ID rather
than per class. Each annotation ID is treated as the statistical unit.

The loader:
1. Opens a GUI to select the parent folder.
2. Finds JSON ground-truth files first.
3. Matches each COCO image to same-stem ENVI .hdr and .bip files.
4. Converts COCO polygon annotations to masks.
5. Randomly samples at most max_spectra_per_id pixels from each annotation.
6. Opens each BIP datacube as a memory map.
7. Reads only selected pixel spectra into RAM.
8. Returns class and provenance information for every spectrum.

Returned arrays
---------------
spectral_data : shape (n_spectra, n_bands)
classes       : shape (n_spectra,)
wavelength    : shape (n_bands,)
cube_names    : shape (n_spectra,)
image_id      : shape (n_spectra,)
annotation_id : shape (n_spectra,)

Required packages
-----------------
pip install numpy pillow spectral

Expected files
--------------
Mix1.json
Mix1.hdr
Mix1.bip
"""

from __future__ import annotations

import json
import warnings
from collections import defaultdict
from pathlib import Path
from typing import Any
import numpy as np
from PIL import Image, ImageDraw
from spectral.io import envi


def browse_data_folder() -> str:
    """
    Open a GUI for selecting the parent folder.

    Returns
    -------
    str
        Selected folder path, or an empty string if the user cancels.
    """
    import tkinter as tk
    from tkinter import filedialog

    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)

    folder_path = filedialog.askdirectory(
        title="Select data folder"
    )

    root.destroy()

    return folder_path


def _polygon_to_mask(
    segmentation: list,
    height: int,
    width: int,
) -> np.ndarray:
    """
    Convert COCO polygon segmentation to a writable Boolean mask.

    Parameters
    ----------
    segmentation : list
        COCO polygon segmentation.

    height : int
        Image height in pixels.

    width : int
        Image width in pixels.

    Returns
    -------
    np.ndarray
        Boolean mask with shape (height, width).
    """
    mask_image = Image.new(
        "1",
        (width, height),
        0,
    )

    drawer = ImageDraw.Draw(mask_image)

    if not segmentation:
        return np.zeros(
            (height, width),
            dtype=bool,
        )

    if isinstance(
        segmentation[0],
        (int, float),
    ):
        polygons = [segmentation]

    else:
        polygons = segmentation

    for polygon in polygons:

        if polygon is None:
            continue

        if len(polygon) < 6:
            continue

        points_array = np.asarray(
            polygon,
            dtype=np.float64,
        ).reshape(-1, 2)

        points = [
            (
                float(x),
                float(y),
            )
            for x, y in points_array
        ]

        drawer.polygon(
            points,
            outline=1,
            fill=1,
        )

    # np.asarray(PIL_image) may produce a read-only array.
    # np.array(..., copy=True) guarantees a writable array.
    return np.array(
        mask_image,
        dtype=bool,
        copy=True,
    )


def _read_wavelength(
    metadata: dict[str, Any],
) -> np.ndarray:
    """
    Read wavelength values from ENVI header metadata.

    Parameters
    ----------
    metadata : dict
        ENVI metadata dictionary.

    Returns
    -------
    np.ndarray
        One-dimensional wavelength array.
    """
    wavelength_value = None

    for key, value in metadata.items():

        key_lower = str(
            key
        ).strip().lower()

        if key_lower in {
            "wavelength",
            "wavelengths",
        }:
            wavelength_value = value
            break

    if wavelength_value is None:
        raise ValueError(
            "The ENVI header does not contain wavelength metadata."
        )

    if isinstance(
        wavelength_value,
        str,
    ):
        wavelength_value = (
            wavelength_value
            .strip()
            .strip("{}")
            .replace("\n", " ")
            .split(",")
        )

    try:
        wavelength = np.asarray(
            wavelength_value,
            dtype=np.float64,
        ).reshape(-1)

    except (
        TypeError,
        ValueError,
    ) as exc:
        raise ValueError(
            "Could not convert wavelength metadata to numeric values."
        ) from exc

    return wavelength


def _validate_coco_image_shape(
    image_info: dict[str, Any],
    hsi_image: Any,
    cube_name: str,
) -> None:
    """
    Confirm that COCO image dimensions agree with the ENVI datacube.
    """
    coco_height = int(
        image_info["height"]
    )

    coco_width = int(
        image_info["width"]
    )

    envi_height = int(
        hsi_image.nrows
    )

    envi_width = int(
        hsi_image.ncols
    )

    if (
        coco_height,
        coco_width,
    ) != (
        envi_height,
        envi_width,
    ):
        raise ValueError(
            f"Spatial-size mismatch for '{cube_name}': "
            f"COCO={(coco_height, coco_width)}, "
            f"ENVI={(envi_height, envi_width)}."
        )


def _sample_annotation_pixels(
    rows: np.ndarray,
    columns: np.ndarray,
    max_spectra_per_id: int | None,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Randomly sample pixel coordinates from one annotation.

    Parameters
    ----------
    rows : np.ndarray
        Candidate row coordinates.

    columns : np.ndarray
        Candidate column coordinates.

    max_spectra_per_id : int or None
        Maximum number of pixels retained for this annotation.

        If None, all annotated pixels are retained.

    rng : np.random.Generator
        Random-number generator.

    Returns
    -------
    sampled_rows : np.ndarray
        Selected row coordinates.

    sampled_columns : np.ndarray
        Selected column coordinates.
    """
    number_of_pixels = rows.size

    if number_of_pixels == 0:
        return rows, columns

    if max_spectra_per_id is None:
        return rows, columns

    if number_of_pixels <= max_spectra_per_id:
        return rows, columns

    selected_indices = rng.choice(
        number_of_pixels,
        size=max_spectra_per_id,
        replace=False,
    )

    sampled_rows = rows[
        selected_indices
    ]

    sampled_columns = columns[
        selected_indices
    ]

    return (
        sampled_rows,
        sampled_columns,
    )


def bip_spectra_loading_by_gt(
    folder_path: str | Path,
    max_spectra_per_id: int | None = 1000,
    random_state: int | None = 42,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    """
    Load spectra from BIP ENVI datacubes using COCO ground truth.

    Each COCO annotation ID is treated as one statistical unit.
    At most max_spectra_per_id pixels are randomly selected from each
    annotation.

    The complete datacube is not loaded into RAM. Each datacube is opened
    as a native BIP memory map, and only selected pixel spectra are copied.

    Parameters
    ----------
    folder_path : str or pathlib.Path
        Parent folder containing matching JSON, HDR, and BIP files.

    max_spectra_per_id : int or None, default=1000
        Maximum number of pixel spectra retained from each annotation ID.

        If an annotation contains fewer pixels, all available pixels are
        retained.

        If None, all annotated pixels are retained.

    random_state : int or None, default=42
        Random seed for repeatable pixel sampling.

        Set to None for non-reproducible random sampling.

    Returns
    -------
    spectral_data : np.ndarray
        Two-dimensional float32 array with shape:

        (number_of_selected_pixels, number_of_bands)

    classes : np.ndarray
        One-dimensional object array containing the class name for every
        row of spectral_data.

    wavelength : np.ndarray
        One-dimensional float64 wavelength array.

    cube_names : np.ndarray
        One-dimensional object array containing the datacube stem for
        every spectrum.

    image_id : np.ndarray
        One-dimensional integer array containing the COCO image ID for
        every spectrum.

    annotation_id : np.ndarray
        One-dimensional integer array containing the COCO annotation ID
        for every spectrum.

    Notes
    -----
    The returned sample-level arrays satisfy:

        spectral_data.shape[0] == classes.shape[0]
        spectral_data.shape[0] == cube_names.shape[0]
        spectral_data.shape[0] == image_id.shape[0]
        spectral_data.shape[0] == annotation_id.shape[0]

    Class balancing is intentionally not performed inside this function.
    It can be performed later using classes, image_id, and annotation_id.

    Machine-learning train/validation/test splitting should be grouped by
    annotation ID or by a combined identifier such as:

        cube_name + image_id + annotation_id

    This prevents pixels from the same statistical unit from appearing in
    different dataset partitions.
    """
    folder_path = Path(
        folder_path
    ).expanduser().resolve()

    if not folder_path.exists():
        raise FileNotFoundError(
            f"Folder does not exist:\n{folder_path}"
        )

    if not folder_path.is_dir():
        raise NotADirectoryError(
            f"The supplied path is not a folder:\n{folder_path}"
        )

    if max_spectra_per_id is not None:

        if not isinstance(
            max_spectra_per_id,
            int,
        ):
            raise TypeError(
                "max_spectra_per_id must be an integer or None."
            )

        if max_spectra_per_id <= 0:
            raise ValueError(
                "max_spectra_per_id must be greater than zero."
            )

    rng = np.random.default_rng(
        random_state
    )

    json_files = sorted(
        folder_path.glob("*.json")
    )

    if not json_files:
        raise FileNotFoundError(
            f"No JSON ground-truth files were found in:\n"
            f"{folder_path}"
        )

    # Each selected-coordinate record contains:
    #
    # (
    #     hdr_path,
    #     bip_path,
    #     cube_name,
    #     class_name,
    #     image_id,
    #     annotation_id,
    #     row,
    #     column,
    # )
    selected_coordinate_records: list[
        tuple[
            str,
            str,
            str,
            str,
            int,
            int,
            int,
            int,
        ]
    ] = []

    # Keep image metadata for validating each cube later.
    coco_image_info_by_cube: dict[
        tuple[str, str],
        dict[str, Any],
    ] = {}

    total_candidate_pixels = 0
    total_selected_pixels = 0
    total_annotations = 0

    print(
        "PASS 1: Reading COCO files and sampling pixels per annotation ID"
    )
    print(
        "---------------------------------------------------------------"
    )

    # ================================================================
    # PASS 1
    # Read JSON files and select pixels from each annotation.
    # No spectral values are read during this pass.
    # ================================================================
    for json_path in json_files:

        try:

            with json_path.open(
                "r",
                encoding="utf-8",
            ) as file:

                coco = json.load(
                    file
                )

        except (
            OSError,
            json.JSONDecodeError,
        ) as exc:

            warnings.warn(
                f"Skipping '{json_path.name}': "
                f"could not read JSON. {exc}"
            )

            continue

        categories = {
            category["id"]: str(
                category.get(
                    "name",
                    category["id"],
                )
            )
            for category in coco.get(
                "categories",
                [],
            )
            if "id" in category
        }

        images = coco.get(
            "images",
            [],
        )

        annotations = coco.get(
            "annotations",
            [],
        )

        if not categories:

            warnings.warn(
                f"Skipping '{json_path.name}': "
                "no COCO categories found."
            )

            continue

        if not images:

            warnings.warn(
                f"Skipping '{json_path.name}': "
                "no COCO image entries found."
            )

            continue

        for image_info in images:

            required_fields = {
                "id",
                "file_name",
                "height",
                "width",
            }

            if not required_fields.issubset(
                image_info
            ):

                warnings.warn(
                    f"Skipping an incomplete image entry in "
                    f"'{json_path.name}'."
                )

                continue

            current_image_id = int(
                image_info["id"]
            )

            file_name = str(
                image_info["file_name"]
            )

            # Only the filename stem is used.
            #
            # For example:
            # Mix1.bip -> Mix1
            # Mix1.bsq -> Mix1
            cube_stem = Path(
                file_name
            ).stem

            hdr_path = (
                folder_path
                / f"{cube_stem}.hdr"
            )

            bip_path = (
                folder_path
                / f"{cube_stem}.bip"
            )

            if not hdr_path.exists():

                warnings.warn(
                    f"Skipping '{cube_stem}': "
                    f"missing {hdr_path.name}."
                )

                continue

            if not bip_path.exists():

                warnings.warn(
                    f"Skipping '{cube_stem}': "
                    f"missing {bip_path.name}."
                )

                continue

            # A zero-byte BIP file can still have a valid-looking HDR file,
            # but there is no binary hyperspectral data to memory-map.
            # Skip it here so the rest of the dataset can still be loaded.
            try:
                bip_file_size = bip_path.stat().st_size
            except OSError as exc:
                warnings.warn(
                    f"Skipping '{cube_stem}': could not read the size of "
                    f"{bip_path.name}. {exc}"
                )
                continue

            if bip_file_size == 0:
                warnings.warn(
                    f"Skipping '{cube_stem}': {bip_path.name} is empty "
                    "(0 bytes). Replace or re-export this BIP file if the "
                    "cube should be included."
                )
                continue

            height = int(
                image_info["height"]
            )

            width = int(
                image_info["width"]
            )

            image_annotations = [
                annotation
                for annotation in annotations
                if int(
                    annotation.get(
                        "image_id",
                        -1,
                    )
                ) == current_image_id
            ]

            if not image_annotations:

                warnings.warn(
                    f"Skipping '{cube_stem}': "
                    "no annotations were found for this image."
                )

                continue

            coco_image_info_by_cube[
                (
                    str(hdr_path),
                    str(bip_path),
                )
            ] = image_info

            for annotation in image_annotations:

                category_id = annotation.get(
                    "category_id"
                )

                current_annotation_id = annotation.get(
                    "id"
                )

                if category_id not in categories:

                    warnings.warn(
                        f"Ignoring annotation in "
                        f"'{json_path.name}' with unknown "
                        f"category_id={category_id}."
                    )

                    continue

                if current_annotation_id is None:

                    warnings.warn(
                        f"Ignoring annotation in "
                        f"'{json_path.name}' because it "
                        "has no annotation ID."
                    )

                    continue

                class_name = categories[
                    category_id
                ]

                segmentation = annotation.get(
                    "segmentation",
                    [],
                )

                if not isinstance(
                    segmentation,
                    list,
                ):
                    raise NotImplementedError(
                        f"'{json_path.name}' contains a non-polygon "
                        "COCO segmentation. This version supports "
                        "polygon segmentations only."
                    )

                annotation_mask = _polygon_to_mask(
                    segmentation=segmentation,
                    height=height,
                    width=width,
                )

                rows, columns = np.nonzero(
                    annotation_mask
                )

                candidate_count = int(
                    rows.size
                )

                if candidate_count == 0:

                    warnings.warn(
                        f"Annotation ID {current_annotation_id} "
                        f"in '{json_path.name}' contains no pixels."
                    )

                    continue

                (
                    sampled_rows,
                    sampled_columns,
                ) = _sample_annotation_pixels(
                    rows=rows,
                    columns=columns,
                    max_spectra_per_id=max_spectra_per_id,
                    rng=rng,
                )

                selected_count = int(
                    sampled_rows.size
                )

                total_annotations += 1
                total_candidate_pixels += candidate_count
                total_selected_pixels += selected_count

                selected_coordinate_records.extend(
                    (
                        str(hdr_path),
                        str(bip_path),
                        cube_stem,
                        class_name,
                        current_image_id,
                        int(current_annotation_id),
                        int(row),
                        int(column),
                    )
                    for row, column in zip(
                        sampled_rows,
                        sampled_columns,
                    )
                )

                print(
                    f"{cube_stem:20s} | "
                    f"{class_name:15s} | "
                    f"image_id={current_image_id:4d} | "
                    f"annotation_id="
                    f"{int(current_annotation_id):4d} | "
                    f"candidates={candidate_count:6,d} | "
                    f"selected={selected_count:6,d}"
                )

    if not selected_coordinate_records:
        raise RuntimeError(
            "No annotated pixels were selected from the JSON files."
        )

    # Group selected records by datacube so each cube is opened once.
    coordinates_by_cube: dict[
        tuple[str, str],
        list[
            tuple[
                str,
                str,
                int,
                int,
                int,
                int,
            ]
        ],
    ] = defaultdict(list)

    for (
        hdr_path,
        bip_path,
        cube_name,
        class_name,
        current_image_id,
        current_annotation_id,
        row,
        column,
    ) in selected_coordinate_records:

        coordinates_by_cube[
            (
                hdr_path,
                bip_path,
            )
        ].append(
            (
                cube_name,
                class_name,
                current_image_id,
                current_annotation_id,
                row,
                column,
            )
        )

    print(
        "\nPASS 2: Reading selected spectra from BIP memory maps"
    )
    print(
        "----------------------------------------------------"
    )

    spectral_blocks: list[
        np.ndarray
    ] = []

    class_blocks: list[
        np.ndarray
    ] = []

    cube_name_blocks: list[
        np.ndarray
    ] = []

    image_id_blocks: list[
        np.ndarray
    ] = []

    annotation_id_blocks: list[
        np.ndarray
    ] = []

    reference_wavelength: (
        np.ndarray | None
    ) = None

    reference_number_of_bands: (
        int | None
    ) = None

    # ================================================================
    # PASS 2
    # Open each BIP cube as a memory map and read selected spectra.
    #
    # IMPORTANT:
    # If one cube is inconsistent with its COCO metadata or with the
    # other ENVI cubes, warn and skip only that cube instead of stopping
    # the complete dataset-loading process.
    # ================================================================
    skipped_cubes_pass2: list[str] = []

    for (
        hdr_path,
        bip_path,
    ), selected_pixels in coordinates_by_cube.items():

        cube_file_name = Path(
            bip_path
        ).name

        # ------------------------------------------------------------
        # Open the ENVI cube.
        # ------------------------------------------------------------
        try:
            hsi_image = envi.open(
                hdr_path,
                image=bip_path,
            )
        except Exception as exc:
            warnings.warn(
                f"Skipping '{cube_file_name}': could not open the ENVI "
                f"cube. The HDR/BIP pair may be corrupted, incomplete, "
                f"or inconsistent. {exc}"
            )
            skipped_cubes_pass2.append(cube_file_name)
            continue

        # ------------------------------------------------------------
        # Confirm BIP interleave.
        # ------------------------------------------------------------
        source_interleave = str(
            hsi_image.metadata.get(
                "interleave",
                "",
            )
        ).strip().lower()

        if source_interleave != "bip":
            warnings.warn(
                f"Skipping '{cube_file_name}': dataset inconsistency. "
                f"Expected ENVI interleave='bip', but the header reports "
                f"'{source_interleave}'."
            )
            skipped_cubes_pass2.append(cube_file_name)
            del hsi_image
            continue

        image_info = coco_image_info_by_cube[
            (
                hdr_path,
                bip_path,
            )
        ]

        # ------------------------------------------------------------
        # COCO spatial dimensions must match the ENVI cube.
        # Example of a skipped mismatch:
        # COCO=(229, 580), ENVI=(115, 290)
        # ------------------------------------------------------------
        try:
            _validate_coco_image_shape(
                image_info=image_info,
                hsi_image=hsi_image,
                cube_name=cube_file_name,
            )
        except ValueError as exc:
            warnings.warn(
                f"Skipping '{cube_file_name}': dataset inconsistency. "
                f"{exc}"
            )
            skipped_cubes_pass2.append(cube_file_name)
            del hsi_image
            continue

        # ------------------------------------------------------------
        # Read and validate wavelength metadata.
        # ------------------------------------------------------------
        try:
            current_wavelength = _read_wavelength(
                hsi_image.metadata
            )
        except (TypeError, ValueError) as exc:
            warnings.warn(
                f"Skipping '{cube_file_name}': invalid or missing "
                f"wavelength metadata. {exc}"
            )
            skipped_cubes_pass2.append(cube_file_name)
            del hsi_image
            continue

        if (
            current_wavelength.size
            != int(hsi_image.nbands)
        ):
            warnings.warn(
                f"Skipping '{cube_file_name}': dataset inconsistency. "
                f"Header contains {current_wavelength.size} wavelength "
                f"values but the ENVI cube contains {hsi_image.nbands} "
                f"bands."
            )
            skipped_cubes_pass2.append(cube_file_name)
            del hsi_image
            continue

        # Compare spectral dimensions with the first successfully loaded
        # cube. Do not establish the reference until the current cube has
        # passed every validation and its spectra have been read.
        if reference_wavelength is not None:

            if (
                int(hsi_image.nbands)
                != reference_number_of_bands
            ):
                warnings.warn(
                    f"Skipping '{cube_file_name}': dataset inconsistency. "
                    f"Band count is {hsi_image.nbands}, while previously "
                    f"accepted cubes use {reference_number_of_bands} bands."
                )
                skipped_cubes_pass2.append(cube_file_name)
                del hsi_image
                continue

            if not np.allclose(
                current_wavelength,
                reference_wavelength,
                rtol=1e-6,
                atol=1e-8,
                equal_nan=True,
            ):
                warnings.warn(
                    f"Skipping '{cube_file_name}': dataset inconsistency. "
                    "Its wavelength bands do not match the previously "
                    "accepted cubes."
                )
                skipped_cubes_pass2.append(cube_file_name)
                del hsi_image
                continue

        # ------------------------------------------------------------
        # Create native BIP memory map.
        # ------------------------------------------------------------
        try:
            cube_memmap = hsi_image.open_memmap(
                interleave="source"
            )
        except Exception as exc:
            warnings.warn(
                f"Skipping '{cube_file_name}': could not create a memory "
                f"map. The BIP file may be empty, corrupted, incomplete, "
                f"or inaccessible. {exc}"
            )
            skipped_cubes_pass2.append(cube_file_name)
            del hsi_image
            continue

        if cube_memmap is None:
            warnings.warn(
                f"Skipping '{cube_file_name}': Spectral Python returned "
                "None when creating the BIP memory map. The BIP file may "
                "be empty, corrupted, incomplete, or inaccessible."
            )
            skipped_cubes_pass2.append(cube_file_name)
            del hsi_image
            continue

        expected_shape = (
            int(hsi_image.nrows),
            int(hsi_image.ncols),
            int(hsi_image.nbands),
        )

        if cube_memmap.shape != expected_shape:
            warnings.warn(
                f"Skipping '{cube_file_name}': dataset inconsistency. "
                f"Expected BIP memory-map shape {expected_shape}, but "
                f"received {cube_memmap.shape}."
            )
            skipped_cubes_pass2.append(cube_file_name)
            del cube_memmap
            del hsi_image
            continue

        # ------------------------------------------------------------
        # Convert selected records to arrays.
        # ------------------------------------------------------------
        selected_cube_names = np.asarray(
            [
                record[0]
                for record in selected_pixels
            ],
            dtype=object,
        )

        selected_classes = np.asarray(
            [
                record[1]
                for record in selected_pixels
            ],
            dtype=object,
        )

        selected_image_ids = np.asarray(
            [
                record[2]
                for record in selected_pixels
            ],
            dtype=np.int64,
        )

        selected_annotation_ids = np.asarray(
            [
                record[3]
                for record in selected_pixels
            ],
            dtype=np.int64,
        )

        rows = np.asarray(
            [
                record[4]
                for record in selected_pixels
            ],
            dtype=np.int64,
        )

        columns = np.asarray(
            [
                record[5]
                for record in selected_pixels
            ],
            dtype=np.int64,
        )

        # Ground-truth pixel coordinates must be inside the ENVI cube.
        if (
            np.any(rows < 0)
            or np.any(
                rows >= hsi_image.nrows
            )
        ):
            warnings.warn(
                f"Skipping '{cube_file_name}': dataset inconsistency. "
                "Ground-truth row coordinates fall outside the ENVI cube."
            )
            skipped_cubes_pass2.append(cube_file_name)
            del cube_memmap
            del hsi_image
            continue

        if (
            np.any(columns < 0)
            or np.any(
                columns >= hsi_image.ncols
            )
        ):
            warnings.warn(
                f"Skipping '{cube_file_name}': dataset inconsistency. "
                "Ground-truth column coordinates fall outside the ENVI "
                "cube."
            )
            skipped_cubes_pass2.append(cube_file_name)
            del cube_memmap
            del hsi_image
            continue

        # ------------------------------------------------------------
        # Read only the selected pixel spectra.
        # ------------------------------------------------------------
        try:
            spectra = np.asarray(
                cube_memmap[
                    rows,
                    columns,
                    :,
                ],
                dtype=np.float32,
            )
        except Exception as exc:
            warnings.warn(
                f"Skipping '{cube_file_name}': selected spectra could not "
                f"be read from the BIP cube. {exc}"
            )
            skipped_cubes_pass2.append(cube_file_name)
            del cube_memmap
            del hsi_image
            continue

        # The first cube that survives all checks becomes the spectral
        # reference for the remaining cubes.
        if reference_wavelength is None:
            reference_wavelength = current_wavelength.copy()
            reference_number_of_bands = int(
                hsi_image.nbands
            )

        spectral_blocks.append(
            spectra
        )

        class_blocks.append(
            selected_classes
        )

        cube_name_blocks.append(
            selected_cube_names
        )

        image_id_blocks.append(
            selected_image_ids
        )

        annotation_id_blocks.append(
            selected_annotation_ids
        )

        print(
            f"{cube_file_name:25s} | "
            f"loaded spectra={spectra.shape[0]:,}"
        )

        del cube_memmap
        del hsi_image

    if skipped_cubes_pass2:
        print(
            "\nPASS 2 warning: skipped inconsistent/unreadable cubes"
        )
        print(
            "----------------------------------------------------"
        )
        for skipped_cube in skipped_cubes_pass2:
            print(
                f"SKIPPED: {skipped_cube}"
            )

    if not spectral_blocks:
        raise RuntimeError(
            "No spectra were loaded from the selected coordinates."
        )

    spectral_data = np.vstack(
        spectral_blocks
    ).astype(
        np.float32,
        copy=False,
    )

    classes = np.concatenate(
        class_blocks
    )

    cube_names = np.concatenate(
        cube_name_blocks
    )

    image_id = np.concatenate(
        image_id_blocks
    ).astype(
        np.int64,
        copy=False,
    )

    annotation_id = np.concatenate(
        annotation_id_blocks
    ).astype(
        np.int64,
        copy=False,
    )

    wavelength = np.asarray(
        reference_wavelength,
        dtype=np.float64,
    )

    # Shuffle every sample-level array using the same permutation.
    permutation = rng.permutation(
        spectral_data.shape[0]
    )

    spectral_data = spectral_data[
        permutation
    ]

    classes = classes[
        permutation
    ]

    cube_names = cube_names[
        permutation
    ]

    image_id = image_id[
        permutation
    ]

    annotation_id = annotation_id[
        permutation
    ]

    print(
        "\nLoading completed"
    )
    print(
        "-----------------"
    )

    print(
        f"Annotations processed : "
        f"{total_annotations:,}"
    )

    print(
        f"Candidate pixels      : "
        f"{total_candidate_pixels:,}"
    )

    print(
        f"Selected spectra      : "
        f"{total_selected_pixels:,}"
    )

    print(
        f"Spectral data shape   : "
        f"{spectral_data.shape}"
    )

    print(
        f"Classes shape         : "
        f"{classes.shape}"
    )

    print(
        f"Wavelength shape      : "
        f"{wavelength.shape}"
    )

    print(
        f"Cube names shape      : "
        f"{cube_names.shape}"
    )

    print(
        f"Image ID shape        : "
        f"{image_id.shape}"
    )

    print(
        f"Annotation ID shape   : "
        f"{annotation_id.shape}"
    )

    return (
        spectral_data,
        classes,
        wavelength,
        cube_names,
        image_id,
        annotation_id,
    )


