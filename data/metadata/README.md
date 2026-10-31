# Real labeled dataset manifest

Training reads `data/metadata/labels.csv`. This manifest must be built from verified real observations; the application does not generate labels or relabel existing demo data.

Required columns:

| Column | Meaning |
| --- | --- |
| `image_path` | JPG/JPEG/PNG file path, absolute or relative to the project root |
| `group_id` | Storm/event identifier used to keep all related images together in one split |
| `cyclone_detected` | Explicit `1`/`true` or `0`/`false` target |
| `imd_class` | Exact one of the eight configured IMD class names for cyclone rows; blank for non-cyclone rows |

The training script requires both detection classes, all eight cyclone intensity classes, at least five independent `group_id` values, and all eight classes represented in the training split. These checks intentionally stop training instead of inventing labels. Keep all images from the same cyclone/event under one group ID to prevent train/validation/test leakage.

Wind-to-IMD label assignments must be verified against authoritative source definitions before writing labels. This application does not infer IMD classes from wind measurements.
