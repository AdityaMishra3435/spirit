"""
Utility functions for data manipulation.
"""

import os
import pandas as pd
import re
from pathlib import Path


def clean_column_names(df):
    """
    Clean column names in a DataFrame by removing spaces and standardizing format.

    Args:
        df (pd.DataFrame): DataFrame to clean.

    Returns:
        pd.DataFrame: DataFrame with cleaned column names.
    """
    # Strip whitespace from column names
    df.columns = df.columns.str.strip()

    # Standardize specific column names
    rename_dict = {
        "DATE (MM/DD/YYYY)": "DATE",
        # Add more renaming if needed
    }

    # Apply renaming if columns exist
    for old_name, new_name in rename_dict.items():
        if old_name in df.columns:
            df.rename(columns={old_name: new_name}, inplace=True)

    return df


def add_datetime_column(
    df,
    date_col="DATE",
    time_col="MST",
    output_col="Date_and_time",
    format_str="%Y_%m_%d_%H_%M",
):
    """
    Add a formatted datetime column to DataFrame.

    Args:
        df (pd.DataFrame): DataFrame to modify.
        date_col (str): Column name containing date.
        time_col (str): Column name containing time.
        output_col (str): Output column name.
        format_str (str): Output datetime format.

    Returns:
        pd.DataFrame: DataFrame with added datetime column.
    """
    # Create datetime from date and time columns
    df[output_col] = pd.to_datetime(
        df[date_col] + " " + df[time_col], format="%m/%d/%Y %H:%M"
    )

    # Format the datetime as specified
    df[output_col] = df[output_col].dt.strftime(format_str)

    return df


def convert_sun_flag(df, column="Sun Flag"):
    """
    Convert sun flag column to binary (0 or 1).

    Args:
        df (pd.DataFrame): DataFrame to modify.
        column (str): Column name to convert.

    Returns:
        pd.DataFrame: DataFrame with converted column.
    """
    if column in df.columns:
        df[column] = df[column].apply(lambda x: 0 if x == 0 else 1)
    return df


def add_image_paths(df, date_time_col, image_folder, extension, output_col):
    """
    Add a column with image paths if they exist.

    Args:
        df (pd.DataFrame): DataFrame to modify.
        date_time_col (str): Column with date and time values.
        image_folder (str): Path to image folder.
        extension (str): Image file extension (e.g., 'jpg', 'png').
        output_col (str): Name of output column to add.

    Returns:
        pd.DataFrame: DataFrame with added image path column.
    """

    def find_image(row):
        image_name = f"{row[date_time_col]}.{extension}"
        image_path = os.path.join(image_folder, image_name)

        if os.path.exists(image_path):
            return image_path
        else:
            return None

    df[output_col] = df.apply(find_image, axis=1)
    return df


def remove_rows_with_missing_values(df, subset=None):
    """
    Remove rows with missing values.

    Args:
        df (pd.DataFrame): DataFrame to clean.
        subset (list, optional): List of columns to check for missing values.

    Returns:
        pd.DataFrame: DataFrame with rows removed.
    """
    return df.dropna(subset=subset)
