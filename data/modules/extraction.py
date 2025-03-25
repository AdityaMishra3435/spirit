"""
Module for extracting and organizing image files from downloaded zip contents.
"""

import os
import re
import datetime
import shutil
from pathlib import Path
import sys
sys.path.append("..")
from utils.file_utils import ensure_dir, copy_file, parse_and_format_timestamp


def move_images_to_destination(src_directory, dest_directory, image_extensions=None):
    """
    Move all images from source directory and its subfolders to destination directory.

    Args:
        src_directory (str): Source directory containing date folders.
        dest_directory (str): Destination directory for all images.
        image_extensions (list): List of image extensions to move.

    Returns:
        int: Number of files moved.
    """
    if image_extensions is None:
        image_extensions = [
            "*.jpg",
            "*.jpeg",
            "*.png",
            "*.gif",
            "*.bmp",
            "*.tiff",
            "*.webp",
        ]

    ensure_dir(dest_directory)
    moved_count = 0

    # Get all subdirectories (date folders)
    folders = [
        os.path.join(src_directory, folder)
        for folder in os.listdir(src_directory)
        if os.path.isdir(os.path.join(src_directory, folder))
    ]

    # Process each folder
    for folder in folders:
        for ext in image_extensions:
            # Find all files with matching extension in the folder
            import glob

            images = glob.glob(os.path.join(folder, ext))

            # Move each image to destination
            for image in images:
                try:
                    dest_path = os.path.join(dest_directory, os.path.basename(image))
                    shutil.move(image, dest_path)
                    moved_count += 1
                except Exception as e:
                    print(f"Error moving {image}: {e}")

    print(f"Moved {moved_count} images to {dest_directory}")
    return moved_count


def organize_images_by_type(
    source_directory,
    raw_dir,
    processed_dir,
    raw_pattern=r"^(\d{14})_11\.jpg$",
    processed_pattern=r"^(\d{14})_1112_BRBG\.png$",
):
    """
    Organize images from source into raw and processed directories based on filename patterns.

    Args:
        source_directory (str): Directory containing all images.
        raw_dir (str): Directory for raw images.
        processed_dir (str): Directory for processed images.
        raw_pattern (str): Regex pattern for raw images.
        processed_pattern (str): Regex pattern for processed images.

    Returns:
        tuple: (raw_count, processed_count) of images organized.
    """
    # Compile regex patterns
    raw_pattern_compiled = re.compile(raw_pattern)
    processed_pattern_compiled = re.compile(processed_pattern)

    # Ensure directories exist
    ensure_dir(raw_dir)
    ensure_dir(processed_dir)

    raw_count = 0
    processed_count = 0

    # Iterate through all files in source
    for filename in os.listdir(source_directory):
        source_path = os.path.join(source_directory, filename)

        # Skip if not a file
        if not os.path.isfile(source_path):
            continue

        # Check if it's a raw image
        if raw_pattern_compiled.match(filename):
            match = raw_pattern_compiled.match(filename)
            timestamp = match.group(1)

            # Format filename
            formatted_filename = parse_and_format_timestamp(
                timestamp, "%Y%m%d%H%M%S", "%Y_%m_%d_%H_%M.jpg"
            )

            # Copy to raw dir
            dest_path = os.path.join(raw_dir, formatted_filename)
            if copy_file(source_path, dest_path):
                raw_count += 1

        # Check if it's a processed image
        elif processed_pattern_compiled.match(filename):
            match = processed_pattern_compiled.match(filename)
            timestamp = match.group(1)

            # Format filename
            formatted_filename = parse_and_format_timestamp(
                timestamp, "%Y%m%d%H%M%S", "%Y_%m_%d_%H_%M.png"
            )

            # Copy to processed dir
            dest_path = os.path.join(processed_dir, formatted_filename)
            if copy_file(source_path, dest_path):
                processed_count += 1

    print(f"Organized {raw_count} raw images and {processed_count} processed images")
    return raw_count, processed_count
