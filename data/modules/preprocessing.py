"""
Module for preprocessing CSV data and merging with image information.
"""

import os
import glob
import datetime
import numpy as np
import os
import pandas as pd
from pathlib import Path
import sys
sys.path.append("..")
from utils.data_utils import (
    clean_column_names,
    add_datetime_column,
    convert_sun_flag,
    add_image_paths,
    remove_rows_with_missing_values,
)


def preprocess_csv(csv_path, output_path=None):
    """
    Preprocess a single CSV file by standardizing column names and adding datetime column.

    Args:
        csv_path (str): Path to CSV file.
        output_path (str, optional): Path to save processed CSV. If None, original is overwritten.

    Returns:
        pd.DataFrame: Processed DataFrame.
    """
    # Read CSV file
    df = pd.read_csv(csv_path)

    # Clean column names
    df = clean_column_names(df)

    # Add datetime column
    df = add_datetime_column(df)

    # Save processed DataFrame
    if output_path:
        df.to_csv(output_path, index=False)
        print(f"Preprocessed CSV saved to {output_path}")
    else:
        df.to_csv(csv_path, index=False)
        print(f"Preprocessed CSV saved back to {csv_path}")

    return df


def merge_csv_with_cloud_data(main_csv_path, cloud_data_csv_path, output_path=None):
    """
    Merge main CSV data with cloud cover and sun flag data.

    Args:
        main_csv_path (str): Path to main CSV file.
        cloud_data_csv_path (str): Path to cloud data CSV file.
        output_path (str, optional): Path to save merged CSV.

    Returns:
        pd.DataFrame: Merged DataFrame.
    """
    # Read CSVs
    df_main = pd.read_csv(main_csv_path)
    df_cloud = pd.read_csv(cloud_data_csv_path)

    # Clean column names
    df_main = clean_column_names(df_main)
    df_cloud = clean_column_names(df_cloud)

    # Add datetime column to both if not already present
    if "Date_and_time" not in df_main.columns:
        df_main = add_datetime_column(df_main)

    if "Date_and_time" not in df_cloud.columns:
        df_cloud = add_datetime_column(df_cloud)

    # Merge on Date_and_time
    df_merged = pd.merge(
        df_main,
        df_cloud[["Date_and_time", "BRBG Total Cloud Cover [%]", "Sun Flag"]],
        on="Date_and_time",
        how="left",
    )

    # Convert non-zero Sun Flag values to 1
    df_merged = convert_sun_flag(df_merged)

    # Rename columns for consistency
    df_merged.rename(
        columns={
            "BRBG Total Cloud Cover [%]": "cloud_obfuscation",
            "Sun Flag": "sun_visibility",
        },
        inplace=True,
    )

    # Save merged data
    if output_path:
        df_merged.to_csv(output_path, index=False)
        print(f"Merged CSV saved to {output_path}")

    return df_merged


def add_image_paths_to_csv(
    csv_path, raw_images_dir, processed_images_dir, output_path=None
):
    """
    Add image paths to CSV if images exist for the timestamps.

    Args:
        csv_path (str): Path to CSV file.
        raw_images_dir (str): Directory with raw images.
        processed_images_dir (str): Directory with processed images.
        output_path (str, optional): Path to save updated CSV.

    Returns:
        pd.DataFrame: DataFrame with image paths.
    """
    # Read CSV
    df = pd.read_csv(csv_path)

    # Add raw image paths
    df = add_image_paths(df, "Date_and_time", raw_images_dir, "jpg", "Raw_images")

    # Add processed image paths
    df = add_image_paths(
        df, "Date_and_time", processed_images_dir, "png", "Processed_Images"
    )

    # Save updated dataframe
    if output_path:
        df.to_csv(output_path, index=False)
        print(f"CSV with image paths saved to {output_path}")
    else:
        df.to_csv(csv_path, index=False)
        print(f"CSV with image paths saved back to {csv_path}")

    return df


def clean_final_dataset(csv_path, output_path=None, required_columns=None):
    """
    Create clean final dataset by removing rows with missing values.

    Args:
        csv_path (str): Path to CSV file.
        output_path (str, optional): Path to save cleaned CSV.
        required_columns (list, optional): Columns that must have values.

    Returns:
        pd.DataFrame: Cleaned DataFrame.
    """
    # Read CSV
    df = pd.read_csv(csv_path)

    # Default required columns if none provided
    if required_columns is None:
        required_columns = ["Raw_images", "Processed_Images"]

    # Remove rows with missing values in required columns
    df_clean = remove_rows_with_missing_values(df, subset=required_columns)

    # Save cleaned DataFrame
    if output_path:
        df_clean.to_csv(output_path, index=False)
        print(f"Cleaned dataset saved to {output_path}")
    else:
        df_clean.to_csv(csv_path, index=False)
        print(f"Cleaned dataset saved back to {csv_path}")

    return df_clean


def create_initial_csv_files(
    main_csv_path,
    cloud_csv_path,
    raw_images_dir,
    processed_images_dir,
    start_date,
    end_date,
):
    """
    Create initial CSV files if they don't exist.

    Args:
        main_csv_path (str): Path to main CSV file.
        cloud_csv_path (str): Path to cloud data CSV file.
        raw_images_dir (str): Directory with raw images.
        processed_images_dir (str): Directory with processed images.
        start_date (datetime.date): Start date for data.
        end_date (datetime.date): End date for data.

    Returns:
        tuple: (main_df, cloud_df) DataFrames created.
    """
    # Get all image files
    raw_images = glob.glob(os.path.join(raw_images_dir, "*.jpg"))
    processed_images = glob.glob(os.path.join(processed_images_dir, "*.png"))

    # Parse dates from filenames
    date_times = []
    raw_image_paths = []
    processed_image_paths = []

    for raw_path in raw_images:
        filename = os.path.basename(raw_path)
        date_time_str = os.path.splitext(filename)[0]  # Remove extension

        # Add to lists if we have both raw and processed image
        processed_path = os.path.join(processed_images_dir, date_time_str + ".png")
        if os.path.exists(processed_path):
            date_times.append(date_time_str)
            raw_image_paths.append(raw_path)
            processed_image_paths.append(processed_path)

    # Create DataFrames
    if not os.path.exists(main_csv_path):
        print(f"Creating initial main CSV file at {main_csv_path}")
        # Convert date_time strings to components
        dates = []
        times = []

        for dt_str in date_times:
            # Format is YYYY_MM_DD_HH_MM
            parts = dt_str.split("_")
            dates.append(f"{parts[1]}/{parts[2]}/{parts[0]}")  # MM/DD/YYYY
            times.append(f"{parts[3]}:{parts[4]}")  # HH:MM

        # Create sample data
        n_samples = len(date_times)

        main_data = {
            "DATE": dates,
            "MST": times,
            "Date_and_time": date_times,
            "Global_horizontal_irradiance": np.random.uniform(0, 1000, n_samples),
            "Direct_normal_irradiance": np.random.uniform(0, 900, n_samples),
            "Diffuse_horizontal_irradiance": np.random.uniform(0, 300, n_samples),
            "Zenith_angle": np.random.uniform(0, 90, n_samples),
            "Azimuth_angle": np.random.uniform(0, 360, n_samples),
            "Raw_images": raw_image_paths,
            "Processed_Images": processed_image_paths,
        }

        main_df = pd.DataFrame(main_data)

        # Save to CSV
        os.makedirs(os.path.dirname(main_csv_path), exist_ok=True)
        main_df.to_csv(main_csv_path, index=False)
    else:
        main_df = pd.read_csv(main_csv_path)

    # Create cloud data CSV if it doesn't exist
    if not os.path.exists(cloud_csv_path):
        print(f"Creating initial cloud data CSV file at {cloud_csv_path}")

        cloud_data = {
            "DATE": main_df["DATE"],
            "MST": main_df["MST"],
            "Date_and_time": main_df["Date_and_time"],
            "BRBG Total Cloud Cover [%]": np.random.uniform(0, 100, len(main_df)),
            "Sun Flag": np.random.choice([0, 1], len(main_df)),
        }

        cloud_df = pd.DataFrame(cloud_data)

        # Save to CSV
        os.makedirs(os.path.dirname(cloud_csv_path), exist_ok=True)
        cloud_df.to_csv(cloud_csv_path, index=False)
    else:
        cloud_df = pd.read_csv(cloud_csv_path)

    return main_df, cloud_df
