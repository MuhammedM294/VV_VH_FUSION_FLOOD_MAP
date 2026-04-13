import numpy as np
import rasterio

def fusion_rule_based(flood_vv: np.ndarray,
                      flood_vh: np.ndarray) -> np.ndarray:
    """
    Deterministic SAR-based fusion.

    """

    merged = np.zeros_like(flood_vv, dtype=np.uint8)

    # Open Water
    merged[(flood_vv == 1) & (flood_vh == 1)] = 1

    # Flooded Vegetation
    merged[(flood_vh == 1) & (flood_vv != 1)] = 2

    return merged


def fusion_vv_dominant(flood_vv, flood_vh):

    merged = np.zeros_like(flood_vv, dtype=np.uint8)

    # Open Water
    merged[flood_vv == 1] = 1

    # Flooded Vegetation
    merged[(flood_vh == 1) & (flood_vv != 1)] = 2

    return merged


def fusion_confidence_weighted(
    flood_vv: np.ndarray,
    flood_vh: np.ndarray,
    prob_vv: np.ndarray,
    prob_vh: np.ndarray,
    t_high: float = 0.7,
    t_low: float = 0.3,
    nodata_class: int | None = None,  # e.g. 255 to explicitly mark pixels with no valid VV+VH
) -> np.ndarray:
    """
    Confidence-weighted deterministic VV/VH fusion (3-class).
    """

    merged = np.zeros_like(flood_vv, dtype=np.uint8)

    # --- Validity comes ONLY from the decision (flood) maps ---
    vv_valid = (flood_vv == 0) | (flood_vv == 1)
    vh_valid = (flood_vh == 0) | (flood_vh == 1)

    # --- Decision masks ---
    vv_is_flood = vv_valid & (flood_vv == 1)
    vv_is_nonflood = vv_valid & (flood_vv == 0)

    vh_is_flood = vh_valid & (flood_vh == 1)

    # --- Confidence masks (only meaningful where the decision exists) ---
    vv_hi = vv_is_flood & (prob_vv > t_high)
    vh_hi = vh_is_flood & (prob_vh > t_high)

    vv_lo = vv_is_nonflood & (prob_vv < t_low)

    # --- Class 1: Open Water (both flood + both confident) ---
    open_mask = vv_hi & vh_hi
    merged[open_mask] = 1

    # --- Class 2: Flooded Vegetation ---
    # VH must be confident flood; VV must not contradict: either missing OR confidently non-flood
    vv_allows_veg = (~vv_valid) | vv_lo
    veg_mask = vh_hi & vv_allows_veg & (~open_mask)
    merged[veg_mask] = 2

    # --- Optional nodata class: neither sensor provides a valid decision ---
    if nodata_class is not None:
        merged[(~vv_valid) & (~vh_valid)] = np.uint8(nodata_class)

    return merged

def save_merged_map(output_path, merged_map, meta):

    meta_out = meta.copy()
    meta_out.update(dtype="uint8", count=1)

    with rasterio.open(output_path, "w", **meta_out) as dst:
        dst.write(merged_map, 1)