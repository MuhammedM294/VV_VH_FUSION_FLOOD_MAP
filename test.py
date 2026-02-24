from pathlib import Path
from typing import Dict, Tuple

import rasterio
import numpy as np
from tqdm import tqdm

# -----------------------------------------------------------------------------
# File discovery
# -----------------------------------------------------------------------------
def get_corresponding_file_paths(flood_vv: Path) -> Dict[str, Path]:
    """
    Given a VV flood map path, derive all corresponding VV/VH products
    and the ESA WorldCover landcover raster in the same directory.
    """
    parent = flood_vv.parent

    paths = {
        # SAR flood maps
        "flood_vv": flood_vv,
        "flood_vh": parent / flood_vv.name.replace("VV", "VH"),

        # Uncertainty
        "uncert_vv": parent / flood_vv.name.replace("FLOOD", "UNCERT"),
        "uncert_vh": parent / flood_vv.name.replace("FLOOD", "UNCERT").replace("VV", "VH"),

        # Likelihood
        "flood_likelihood_vv": parent / flood_vv.name.replace("FLOOD", "FLOOD_LIKELIHOOD"),
        "flood_likelihood_vh": parent / flood_vv.name.replace("FLOOD", "FLOOD_LIKELIHOOD").replace("VV", "VH"),

        # Probability
        "flood_probability_vv": parent / flood_vv.name.replace("FLOOD", "FLOOD_PROBABILITY"),
        "flood_probability_vh": parent / flood_vv.name.replace("FLOOD", "FLOOD_PROBABILITY").replace("VV", "VH"),
    }

    # --- Landcover (ESA WorldCover) ---
    lc_candidates = list(parent.glob("ESA-WorldCover-2021-V200*.tif"))
    if len(lc_candidates) != 1:
        raise RuntimeError(
            f"Expected exactly one ESA WorldCover file in {parent}, "
            f"found {len(lc_candidates)}"
        )
    paths["landcover"] = lc_candidates[0]

    # --- Existence check ---
    for key, path in paths.items():
        if not path.exists():
            raise FileNotFoundError(f"[{key}] File does not exist: {path}")

    return paths

# -----------------------------------------------------------------------------
# Raster reading
# -----------------------------------------------------------------------------
def read_rasters(paths: Dict[str, Path]) -> Tuple[Dict[str, np.ndarray], dict]:
    """
    Read all rasters into memory and return arrays + metadata
    (metadata taken from flood_vv as reference).
    """
    arrays: Dict[str, np.ndarray] = {}

    # Reference metadata
    with rasterio.open(paths["flood_vv"]) as src:
        arrays["flood_vv"] = src.read(1)
        meta = src.meta.copy()

    # Read the rest
    for key, path in paths.items():
        if key == "flood_vv":
            continue
        with rasterio.open(path) as src:
            arrays[key] = src.read(1)

    return arrays, meta

# -----------------------------------------------------------------------------
# Validation utilities
# -----------------------------------------------------------------------------
def validate_arrays(arrays: Dict[str, np.ndarray]) -> None:
    """
    Ensure all required arrays exist and have consistent shapes.
    """
    required_keys = {
        "flood_vv",
        "flood_vh",
        "uncert_vv",
        "uncert_vh",
        "flood_likelihood_vv",
        "flood_likelihood_vh",
        "flood_probability_vv",
        "flood_probability_vh",
        "landcover",
    }

    missing = required_keys - arrays.keys()
    if missing:
        raise KeyError(f"Missing arrays: {missing}")

    ref_shape = arrays["flood_vv"].shape
    for key, arr in arrays.items():
        if arr.shape != ref_shape:
            raise ValueError(f"Shape mismatch for {key}: {arr.shape} vs {ref_shape}")

def inspect_landcover(landcover: np.ndarray) -> None:
    """
    Quick sanity check for ESA WorldCover classes.
    """
    valid_codes = {10, 20, 30, 40, 50, 60, 70, 80, 90, 95, 100}
    present = set(np.unique(landcover))

    unexpected = present - valid_codes
    if unexpected:
        print(f"⚠️ Unexpected landcover codes detected: {unexpected}")
    else:
        print("✓ ESA WorldCover classes look valid")

# -----------------------------------------------------------------------------
# Fusion
# -----------------------------------------------------------------------------
def fuse_probability_landcover(
    flood_vv: np.ndarray,
    flood_vh: np.ndarray,
    prob_vv: np.ndarray,
    prob_vh: np.ndarray,
    landcover: np.ndarray,
    nodata=np.nan,
):
    """
    Landcover-aware flood probability fusion producing multi-class flood map.

    Classes in fused_flood:
        0: non-flood
        1: VV/VH agreement-based fusion
        2: VH-only dominant class
        3: VV-only flood
        4: VH-only flood
    """

    # Initialize outputs
    fused_prob = np.full_like(prob_vv, nodata, dtype=np.float32)
    fused_flood = np.zeros_like(flood_vv, dtype=np.uint8)

    # --- VH-only dominant classes ---
    VH_ONLY = {10, 20, 30,40, 90, 95}
   

    # --- Validity masks ---
    vv_valid = ~np.isnan(prob_vv)
    vh_valid = ~np.isnan(prob_vh)
    both_valid = vv_valid & vh_valid

    # 1️⃣ VH-only dominant classes (only where flood_vh == 1)
    vh_mask = np.isin(landcover, list(VH_ONLY)) & vh_valid & (flood_vh == 1)
    fused_prob[vh_mask] = prob_vh[vh_mask]
    fused_flood[vh_mask] = 2  # VH-only dominant class

    # Remaining pixels for VV/VH fusion
    remaining = ~vh_mask & both_valid

    # 2️⃣ Agreement-based fusion
    agree = remaining & (flood_vv == 1) & (flood_vh == 1)
    fused_prob[agree] = 0.5 * (prob_vv[agree] + prob_vh[agree])
    fused_flood[agree] = 1  # agreement-based class

    # # 3️⃣ Disagreement → take the smaller probability
    # disagree = remaining & (flood_vv != flood_vh)
    # fused_prob[disagree] = np.minimum(prob_vv[disagree], prob_vh[disagree])
    # # fused_flood remains 0 (non-flood) for disagreement

    # 4️⃣ Fallback (only one channel valid AND flooded)
    only_vv = vv_valid & ~vh_valid & (flood_vv == 1)
    only_vh = vh_valid & ~vv_valid & (flood_vh == 1)
    fused_prob[only_vv] = prob_vv[only_vv]
    fused_prob[only_vh] = prob_vh[only_vh]
    fused_flood[only_vv] = 3  # VV-only flood
    fused_flood[only_vh] = 4  # VH-only flood

    return fused_flood, fused_prob


# -----------------------------------------------------------------------------
# Main execution
# -----------------------------------------------------------------------------
def main() -> None:
    base_dir = Path(
        "/eodc/private/tuwgeo/users/mabdelaa/repos/vv_vh_fusion/data/EQUI7_EU020M"
    )

    vv_flood_maps = [
        img
        for img in base_dir.glob("*/*.tif")
        if "FLOOD-EXPF" in img.name and "_VV_" in img.name
    ]

    print(f"Found {len(vv_flood_maps)} VV flood maps")

    save_path = base_dir / "Fusion/Landcover"
    save_path.mkdir(parents=True, exist_ok=True)

    for flood_vv in tqdm(vv_flood_maps, desc="Processing scenes"):
        print(f"\nProcessing: {flood_vv.name}")

        paths = get_corresponding_file_paths(flood_vv)
        arrays, meta = read_rasters(paths)

        validate_arrays(arrays)
        inspect_landcover(arrays["landcover"])

        fused_flood, fused_likelihood = fuse_probability_landcover(
            arrays["flood_vv"],
            arrays["flood_vh"],
            arrays["flood_likelihood_vv"],
            arrays["flood_likelihood_vh"],
            arrays["landcover"]
        )

        fused_flood_path = save_path / flood_vv.name.replace(
            "FLOOD-EXPF", "FUSED-FLOOD-EXPF"
        ).replace("VV", "FUSED")
        fused_likelihood_path = save_path / flood_vv.name.replace(
            "FLOOD-EXPF", "FUSED-FLOOD-LIKELIHOOD"
        ).replace("VV", "FUSED")

        # Update metadata types
        meta_flood = meta.copy()
        meta_flood.update(dtype=rasterio.uint8, nodata=0)
        meta_prob = meta.copy()
        meta_prob.update(dtype=rasterio.float32, nodata=np.nan)

        with rasterio.open(fused_flood_path, "w", **meta_flood) as dst:
            dst.write(fused_flood, 1)
        with rasterio.open(fused_likelihood_path, "w", **meta_prob) as dst:
            dst.write(fused_likelihood, 1)

        print(f"Saved fused flood: {fused_flood_path.name}")
        print(f"Saved fused likelihood: {fused_likelihood_path.name}")

if __name__ == "__main__":
    main()
