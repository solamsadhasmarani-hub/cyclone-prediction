from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.prediction import ModelUnavailableError, predict_image  # noqa: E402
from utils.preprocessing import load_rgb_image  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Run a trained CYCLO SENSE MobileNetV2 model.")
    parser.add_argument("image", type=Path, help="JPG, JPEG, or PNG image path")
    args = parser.parse_args()
    if args.image.suffix.casefold() not in {".jpg", ".jpeg", ".png"}:
        parser.error("image must be JPG, JPEG, or PNG")
    try:
        load_rgb_image(args.image)
        result = predict_image(args.image)
    except (ModelUnavailableError, ValueError) as error:
        print(f"Prediction unavailable: {error}", file=sys.stderr)
        return 1
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
