from pathlib import Path
from tqdm import tqdm
import rasterio

from vv_vh_fusion.io.loader import get_corresponding_file_paths
from vv_vh_fusion.io.reader import read_rasters
from vv_vh_fusion.fusion.deterministic import (
    fusion_rule_based,
    fusion_vv_dominant,
    fusion_confidence_weighted,
)


def _make_output_name(img: Path, method_name: str) -> str:
    """
    Build output filename that clearly indicates VV+VH fusion.
    Keeps original scene id, removes VV token, appends fusion method.
    """
    stem = img.stem
    # remove VV token safely (covers both -VV and _VV conventions)
    stem = stem.replace("_VV", "").replace("-VV", "")
    return f"{stem}_FUSION-VV-VH-{method_name.upper()}.tif"


def _write_geotiff(output_path: Path, array, meta: dict) -> None:
    meta_out = meta.copy()
    meta_out.update(dtype="uint8", count=1)
    with rasterio.open(output_path, "w", **meta_out) as dst:
        dst.write(array.astype("uint8"), 1)


# ------------------------------------------------------------------
# CONFIG
# ------------------------------------------------------------------

base_dir_path = Path(
    "/eodc/private/tuwgeo/users/mabdelaa/repos/vv_vh_fusion/data/EQUI7_SA020M"
)

output_root = base_dir_path / "fusion_outputs"
output_root.mkdir(exist_ok=True)

# ------------------------------------------------------------------
# COLLECT VV FLOOD FILES
# ------------------------------------------------------------------

imgs = [
    img
    for img in base_dir_path.glob("*/*.tif")
    if "FLOOD-EXPF" in img.name and "VV" in img.name
]

print(f"Found {len(imgs)} VV flood images.")

# ------------------------------------------------------------------
# PROCESS SCENES
# ------------------------------------------------------------------

for img in tqdm(imgs, desc="Running Deterministic Fusion"):

    scene = get_corresponding_file_paths(img)
    arrays, meta = read_rasters(scene)

    flood_vv = arrays["flood_vv"]
    flood_vh = arrays["flood_vh"]

    prob_vv = arrays["flood_probability_vv"]
    prob_vh = arrays["flood_probability_vh"]

    # Run methods
    # Run methods
    results = {
        "rule_based": fusion_rule_based(flood_vv, flood_vh),
        "vv_dominant": fusion_vv_dominant(flood_vv, flood_vh),
        "confidence_weighted": fusion_confidence_weighted(
            flood_vv=flood_vv,
            flood_vh=flood_vh,
            prob_vv=prob_vv,
            prob_vh=prob_vh,
            t_high=0.7,
            t_low=0.3,
        ),
    }

    # Save each method result
    for method_name, merged_map in results.items():
        output_dir = output_root / method_name
        output_dir.mkdir(exist_ok=True)

        output_name = _make_output_name(img, method_name)
        output_path = output_dir / output_name

        _write_geotiff(output_path, merged_map, meta)

print("Deterministic fusion completed.")