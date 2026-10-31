"""Placeholder for a future authoritative wind-to-IMD label mapping."""

from ai.config.imd_classes import IMD_CLASSES

MAPPING_STATUS = "NOT YET VERIFIED"

# Intentionally empty. Do not populate until the source field, units, averaging
# period, category thresholds, and authoritative provenance have been verified.
VERIFIED_WIND_TO_IMD_MAPPING: tuple[dict[str, object], ...] = ()

FUTURE_MAPPING_RECORD_FIELDS = (
    "wind_speed",
    "wind_unit",
    "averaging_period",
    "verified_intensity_category",
    "imd_class_index",
    "imd_class_name",
    "source_field",
    "authoritative_source",
    "verification_reference",
)

IMD_CLASS_NAMES = IMD_CLASSES
