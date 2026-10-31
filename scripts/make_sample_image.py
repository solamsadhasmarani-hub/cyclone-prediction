
import glob
import os

import numpy as np
import xarray as xr
from PIL import Image

files = glob.glob("data/raw/hursat/**/*.nc", recursive=True)

file_path = next(
    (
        path for path in files
        if "NADA" in os.path.basename(path)
        and "HIM-8" in os.path.basename(path)
    ),
    None,
)

if file_path is None:
    raise FileNotFoundError("No matching NADA HIM-8 NetCDF file found.")

with xr.open_dataset(file_path) as dataset:
    values = dataset["IRWIN"].values.squeeze().astype(float)

low, high = np.nanpercentile(values, [2, 98])

if not high > low:
    raise ValueError("Satellite channel has no usable value range.")

normalized = np.clip((values - low) / (high - low), 0, 1)
normalized = np.nan_to_num(normalized, nan=0.0, posinf=1.0, neginf=0.0)

output_path = "data/processed/images/sample/NADA_IRWIN_sample.png"
os.makedirs(os.path.dirname(output_path), exist_ok=True)

image = Image.fromarray((normalized * 255).astype("uint8"))
image.resize((224, 224)).save(output_path)

print("SUCCESS - Image saved:", output_path)
