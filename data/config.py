"""
Configuration settings for solar data processing.
"""

import os
from pathlib import Path

# Base paths
ROOT_DIR = Path(os.path.dirname(os.path.abspath(__file__)))
IMAGE_ZIPS_DIR = ROOT_DIR / "image-zips"
RAW_IMAGES_DIR = ROOT_DIR / "raw_images"
PROCESSED_IMAGES_DIR = ROOT_DIR / "processed_images"
CSV_DIR = ROOT_DIR / "csv"

# URL base for downloading
BASE_URL = "https://midcdmz.nrel.gov/apps/imageranim.pl"

# Regular expressions for image file matching
RAW_IMAGE_PATTERN = r"^(\d{14})_11\.jpg$"
PROCESSED_IMAGE_PATTERN = r"^(\d{14})_1112_BRBG\.png$"

# Dataset settings
DATASET_NAME = "solar_irradiance_dataset"
HF_REPO = "your-username/solar_irradiance_dataset"  # Change to your HF username

# Define the columns to keep in the final dataset
DATASET_COLUMNS = [
    "DATE",
    "MST",
    "Global_horizontal_irradiance",
    "Direct_normal_irradiance",
    "Diffuse_horizontal_irradiance",
    "Zenith_angle",
    "Azimuth_angle",
    "cloud_obfuscation",
    "sun_visibility",
    "Raw_images",
]


# Embedding settings
EMBEDDINGS_DIR = ROOT_DIR / "embeddings"
DEFAULT_MODEL_NAME = "google/vit-base-patch16-224-in21k"
IMAGE_PADDING = (0, 0)  # Default padding (width, height) for images
