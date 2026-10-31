# Model Strategy

The project will start with simple and explainable baselines before moving to more advanced methods.

## Phase 1

- local sample dataset
- metadata validation
- image preprocessing
- split-by-cyclone evaluation setup

## Phase 2

- synthetic cyclone/no-cyclone detection dataset
- ResNet-18 baseline for cyclone detection
- validation metrics and model checkpoint saving
- inference API logic for detecting whether an image contains a cyclone

## Phase 3

- synthetic multi-class classification dataset for cyclone stages
- ResNet-18 baseline for stage classification
- confusion matrix and classification report generation
- inference using saved model weights on new images

The current classifier has four synthetic development-stage labels (`developing`, `mature`, `weakening`, `dissipating`). It is not an eight-class IMD intensity classifier.

## Proposed SIH classification pipeline

- MobileNetV2 with transfer learning and a custom classification head
- 224 × 224 satellite image input
- eight IMD intensity outputs: Low Pressure Area, Depression, Deep Depression, Cyclonic Storm, Severe Cyclonic Storm, Very Severe Cyclonic Storm, Extremely Severe Storm, and Super Cyclonic Storm

This proposed pipeline is not implemented in the current prototype. Dashboard probabilities shown against those classes are illustrative demo data, not model outputs.

## Phase 4

- Grad-CAM-style visualization for model interpretability
- heatmap extraction from the final convolutional feature maps
- explanation output describing the dominant image regions used by the model
- explanations generated only from an explicitly supplied image

## Phase 5

- synthetic intensity forecasting dataset for local demo use only
- Random Forest baseline for cyclone intensity prediction
- MAE and RMSE evaluation on a grouped test split
- output format with current intensity, predicted intensity, and prediction horizon

## Phase 6

- synthetic cyclone track dataset for local demo use only
- Random Forest baseline for future position forecasting
- RMSE evaluation for latitude and longitude predictions
- output format with future latitude, longitude, and prediction horizon

## Planned model progression

1. Cyclone detection using transfer learning with EfficientNet or ResNet
2. Classification into configurable storm stages
3. Explainability using Grad-CAM-style visualization
4. Baseline intensity prediction using Random Forest or XGBoost
5. Baseline track prediction using regression models

## Constraints

- Avoid fabricated labels or real-time meteorological data
- Keep model complexity low until the baseline proves useful
- Use sample-only data in the demo environment
- Do not present the ResNet-18 four-stage prototype as the proposed MobileNetV2 eight-class model
- Do not display evaluation metrics unless supplied by actual model evaluation data
