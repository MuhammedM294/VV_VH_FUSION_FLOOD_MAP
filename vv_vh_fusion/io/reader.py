import rasterio
from typing import Tuple, Dict
import numpy as np


def read_rasters(scene) -> Tuple[Dict[str, np.ndarray], dict]:
    """
    Reads all rasters associated with a S1Images object.
    """

    arrays = {}

    # Use flood_vv as spatial reference
    with rasterio.open(scene.flood_vv) as ref:
        arrays["flood_vv"] = ref.read(1)
        meta = ref.meta.copy()
        ref_shape = ref.shape
        ref_crs = ref.crs
        ref_transform = ref.transform

    # Read remaining rasters and validate alignment
    for key, path in scene.__dict__.items():
        if key == "flood_vv":
            continue

        with rasterio.open(path) as src:
            if src.shape != ref_shape:
                raise ValueError(f"{key} shape mismatch: {src.shape} != {ref_shape}")
            if src.crs != ref_crs:
                raise ValueError(f"{key} CRS mismatch.")
            if src.transform != ref_transform:
                raise ValueError(f"{key} transform mismatch.")

            arrays[key] = src.read(1)

    return arrays, meta