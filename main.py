import numpy as np
import rasterio
from pathlib import Path
from tqdm import tqdm


def generate_flood_likelihood_and_probability(
    uncert_path, likelihood_output, probability_output
):
    flood_file_name = uncert_path.name.replace("UNCERT", "FLOOD")
    flood_path = uncert_path.parent / flood_file_name
    if not flood_path.exists():
        print("Flood file does not exist for:", uncert_path.name)
        return
    with rasterio.open(uncert_path) as src:
        uncert_data = src.read(1)
        meta = src.meta
    with rasterio.open(flood_path) as src:
        flood_data = src.read(1)

    meta["dtype"] = "float32"
    if likelihood_output:
        flood_likelihood = np.where(flood_data == 1, 100 - uncert_data, uncert_data)
        likelihood_output = uncert_path.name.replace("UNCERT", "FLOOD_LIKELIHOOD")
        likelihood_output = uncert_path.parent / likelihood_output
        with rasterio.open(likelihood_output, "w", **meta) as dst:
            dst.write(flood_likelihood, 1)

    if probability_output:
        flood_probability = np.where(
            flood_data == 1, 1 - (uncert_data / 100.0), uncert_data / 100.0
        )
        probability_output = uncert_path.name.replace("UNCERT", "FLOOD_PROBABILITY")
        probability_output = uncert_path.parent / probability_output
        with rasterio.open(probability_output, "w", **meta) as dst:
            dst.write(flood_probability, 1)


if __name__ == "__main__":
    data_path = Path("/eodc/private/tuwgeo/users/mabdelaa/repos/vv_vh_fusion/data")
    uncert_images = [image for image in data_path.glob("*/*/*UNCERT*.tif")]
    print(f"Found {len(uncert_images)} uncertainty images.")
    for uncert_image in tqdm(uncert_images):
        print("Processing:", uncert_image.name)
        generate_flood_likelihood_and_probability(
            uncert_image, likelihood_output=True, probability_output=True
        )
    print("Processing completed.")
