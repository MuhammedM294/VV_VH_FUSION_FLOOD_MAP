from pathlib import Path
from dataclasses import dataclass
from typing import Dict
from tqdm import tqdm

@dataclass
class S1Images:
    flood_vv: Path
    flood_vh: Path
    uncert_vv: Path
    uncert_vh: Path
    flood_likelihood_vv: Path
    flood_likelihood_vh: Path
    flood_probability_vv: Path
    flood_probability_vh: Path
    landcover: Path


def _replace_token(filename: str, old: str, new: str) -> str:
    """Replace token only if present as a distinct component."""
    if old not in filename:
        raise ValueError(f"Token '{old}' not found in filename {filename}")
    return filename.replace(old, new)


def get_corresponding_file_paths(flood_vv: Path) -> S1Images:
    parent = flood_vv.parent
    name = flood_vv.name

    if "VV" not in name or "FLOOD" not in name:
        raise ValueError(f"Input file must be VV FLOOD product. Got {name}")

    # --- SAR-derived products ---
    flood_vh = parent / _replace_token(name, "VV", "VH")

    uncert_vv = parent / _replace_token(name, "FLOOD", "UNCERT")
    uncert_vh = parent / _replace_token(
        _replace_token(name, "FLOOD", "UNCERT"),
        "VV",
        "VH",
    )

    likelihood_vv = parent / _replace_token(name, "FLOOD", "FLOOD_LIKELIHOOD")
    likelihood_vh = parent / _replace_token(
        _replace_token(name, "FLOOD", "FLOOD_LIKELIHOOD"),
        "VV",
        "VH",
    )

    probability_vv = parent / _replace_token(name, "FLOOD", "FLOOD_PROBABILITY")
    probability_vh = parent / _replace_token(
        _replace_token(name, "FLOOD", "FLOOD_PROBABILITY"),
        "VV",
        "VH",
    )

    # --- Land cover ---
    lc_candidates = list(parent.glob("ESA-WorldCover-2021-V200*.tif"))
    if len(lc_candidates) != 1:
        raise FileNotFoundError(
            f"Expected exactly one ESA WorldCover file in {parent}, "
            f"found {len(lc_candidates)}"
        )

    landcover = lc_candidates[0]

    # --- Existence check ---
    files_to_check = [
        flood_vv,
        flood_vh,
        uncert_vv,
        uncert_vh,
        likelihood_vv,
        likelihood_vh,
        probability_vv,
        probability_vh,
        landcover,
    ]

    missing = [str(p) for p in files_to_check if not p.exists()]
    if missing:
        raise FileNotFoundError("Missing files:\n" + "\n".join(missing))

    return S1Images(
        flood_vv=flood_vv,
        flood_vh=flood_vh,
        uncert_vv=uncert_vv,
        uncert_vh=uncert_vh,
        flood_likelihood_vv=likelihood_vv,
        flood_likelihood_vh=likelihood_vh,
        flood_probability_vv=probability_vv,
        flood_probability_vh=probability_vh,
        landcover=landcover,
    )

if __name__=="__main__":

    base_dir_path = Path(
    "/eodc/private/tuwgeo/users/mabdelaa/repos/vv_vh_fusion/data/EQUI7_AS020M"
    )

    imgs = [
        img
        for img in base_dir_path.glob("*/*.tif")
        if "FLOOD-EXPF" in img.name and "VV" in img.name
    ]

    # imgs = imgs[2:]
    for img in tqdm(imgs):
        print(img.name)
        sar_scence = get_corresponding_file_paths(img)
        print(sar_scence.landcover)
        break