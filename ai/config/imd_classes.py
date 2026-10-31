"""Official-order IMD intensity class names for the SIH26070 proposal."""

IMD_CLASSES: tuple[str, ...] = (
    "Low Pressure Area",
    "Depression",
    "Deep Depression",
    "Cyclonic Storm",
    "Severe Cyclonic Storm",
    "Very Severe Cyclonic Storm",
    "Extremely Severe Storm",
    "Super Cyclonic Storm",
)

IMD_CLASS_TO_INDEX: dict[str, int] = {
    class_name: index for index, class_name in enumerate(IMD_CLASSES)
}
