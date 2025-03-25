"""
Module for creating and uploading Hugging Face datasets.
"""

import pandas as pd
import os
from datasets import Dataset, Features, Image, Value
from huggingface_hub import HfApi, HfFolder
from pathlib import Path
import logging


def prepare_dataset_dict(csv_path, columns_to_include=None):
    """
    Prepare data dictionary from CSV for Hugging Face dataset.

    Args:
        csv_path (str): Path to CSV file.
        columns_to_include (list, optional): List of columns to include in dataset.

    Returns:
        dict: Data dictionary for Dataset.from_dict()
    """
    # Read CSV
    df = pd.read_csv(csv_path)

    # If columns to include is specified, filter columns
    if columns_to_include:
        df = df[columns_to_include]

    # Initialize data dictionary
    data = {col: [] for col in df.columns}

    # Populate data dictionary
    for _, row in df.iterrows():
        for col in df.columns:
            data[col].append(row[col])

    return data


def create_dataset_features(data_dict):
    """
    Create features dictionary for Hugging Face dataset.

    Args:
        data_dict (dict): Data dictionary.

    Returns:
        Features: HF Features object.
    """
    feature_dict = {}

    # Determine feature type for each column
    for key in data_dict.keys():
        if key.lower().endswith(("images", "image", "_img", "_path")):
            feature_dict[key] = Image()
        elif isinstance(data_dict[key][0], (int, float)) and not isinstance(
            data_dict[key][0], bool
        ):
            feature_dict[key] = Value("float32")
        else:
            feature_dict[key] = Value("string")

    return Features(feature_dict)


def create_hf_dataset(data_dict, features=None):
    """
    Create Hugging Face dataset from data dictionary.

    Args:
        data_dict (dict): Data dictionary.
        features (Features, optional): HF Features object.

    Returns:
        Dataset: HF Dataset object.
    """
    if features is None:
        features = create_dataset_features(data_dict)

    return Dataset.from_dict(data_dict, features=features)


def upload_to_hf_hub(dataset, repo_name, token=None):
    """
    Upload dataset to Hugging Face Hub.

    Args:
        dataset (Dataset): HF Dataset object.
        repo_name (str): Name of HF repo (username/dataset-name).
        token (str, optional): HF token. If None, looks for token in env or .huggingface folder.

    Returns:
        bool: True if successful.
    """
    try:
        # Save token if provided
        if token:
            HfFolder.save_token(token)

        # Push dataset to hub
        dataset.push_to_hub(repo_name)
        print(f"Dataset successfully uploaded to {repo_name}")
        return True

    except Exception as e:
        print(f"Error uploading dataset to HF Hub: {e}")
        return False


def create_and_upload_dataset(csv_path, repo_name, token=None, columns_to_include=None):
    """
    Create and upload dataset to Hugging Face Hub in one step.

    Args:
        csv_path (str): Path to CSV file.
        repo_name (str): Name of HF repo (username/dataset-name).
        token (str, optional): HF token.
        columns_to_include (list, optional): List of columns to include.

    Returns:
        bool: True if successful.
    """
    try:
        # Prepare data
        data_dict = prepare_dataset_dict(csv_path, columns_to_include)

        # Create features
        features = create_dataset_features(data_dict)

        # Create dataset
        dataset = create_hf_dataset(data_dict, features)

        # Upload to Hub
        return upload_to_hf_hub(dataset, repo_name, token)

    except Exception as e:
        print(f"Error creating and uploading dataset: {e}")
        return False
