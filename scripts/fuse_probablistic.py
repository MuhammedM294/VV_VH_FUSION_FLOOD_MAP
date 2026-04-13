from pathlib import Path
from tqdm import tqdm
import numpy as np
import rasterio

from vv_vh_fusion.io.loader import get_corresponding_file_paths
from vv_vh_fusion.io.reader import read_rasters

from vv_vh_fusion.fusion.probabilistic import (
    fuse_bayesian_product,
    fuse_weighted_logit,
    fuse_entropy_weighted_average,
    probability_to_binary,
)


def _make_stem(img: Path) -> str:
    stem = img.stem.replace("_VV", "").replace("-VV", "")
    return stem


def _write_float(output_path: Path, arr: np.ndarray, meta: dict, nodata=np.nan) -> None:
    meta_out = meta.copy()
    meta_out.update(dtype="float32", count=1, nodata=nodata)
    with rasterio.open(output_path, "w", **meta_out) as dst:
        dst.write(arr.astype("float32"), 1)


def _write_uint8(output_path: Path, arr: np.ndarray, meta: dict, nodata: int | None = None) -> None:
    meta_out = meta.copy()
    meta_out.update(dtype="uint8", count=1)
    if nodata is not None:
        meta_out.update(nodata=nodata)
    with rasterio.open(output_path, "w", **meta_out) as dst:
        dst.write(arr.astype("uint8"), 1)


# ------------------------------------------------------------------
# CONFIG
# ------------------------------------------------------------------
base_dir_path = Path("/eodc/private/tuwgeo/users/mabdelaa/repos/vv_vh_fusion/data/EQUI7_SA020M")
output_root = base_dir_path / "fusion_outputs_group_b"
output_root.mkdir(exist_ok=True)

imgs = [
    img
    for img in base_dir_path.glob("*/*.tif")
    if "FLOOD-EXPF" in img.name and "VV" in img.name
]
print(f"Found {len(imgs)} VV flood images.")

# ------------------------------------------------------------------
# RUN
# ------------------------------------------------------------------
for img in tqdm(imgs, desc="Running Group B (Probabilistic Fusion)"):

    scene = get_corresponding_file_paths(img)
    arrays, meta = read_rasters(scene)

    flood_vv = arrays["flood_vv"]
    flood_vh = arrays["flood_vh"]
    prob_vv = arrays["flood_probability_vv"]
    prob_vh = arrays["flood_probability_vh"]

    stem = _make_stem(img)

    methods = {
        "B1_BAYES_PRODUCT": lambda: fuse_bayesian_product(flood_vv, flood_vh, prob_vv, prob_vh),
        "B2_WEIGHTED_LOGIT": lambda: fuse_weighted_logit(flood_vv, flood_vh, prob_vv, prob_vh, w_vv=0.5, w_vh=0.5),
        "B3_ENTROPY_WEIGHTED": lambda: fuse_entropy_weighted_average(flood_vv, flood_vh, prob_vv, prob_vh),
    }

    for method_name, fn in methods.items():
        out_dir = output_root / method_name
        out_dir.mkdir(exist_ok=True)

        p_fused, valid = fn()

        # enforce "no prob where no real data": set invalid to NaN
        p_fused = p_fused.copy()
        p_fused[~valid] = np.nan

        # write probability
        p_path = out_dir / f"{stem}_FUSION-VV-VH_{method_name}_PFUSED.tif"
        _write_float(p_path, p_fused, meta, nodata=np.nan)

        # write thresholded flood (binary)
        flood_fused = probability_to_binary(p_fused, threshold=0.5)
        # optional: nodata=255 for pixels with no valid sensor
        flood_fused_uint8 = flood_fused.copy()
        flood_fused_uint8[~valid] = 255

        f_path = out_dir / f"{stem}_FUSION-VV-VH_{method_name}_FLOOD.tif"
        _write_uint8(f_path, flood_fused_uint8, meta, nodata=255)

print("Group B probabilistic fusion completed.")