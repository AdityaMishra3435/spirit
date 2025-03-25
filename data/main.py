"""
Main script to execute the full data processing pipeline.
"""

import os
import argparse
import datetime
from pathlib import Path

import config
from utils.file_utils import ensure_dir
from modules.download import download_images_in_range
from modules.extraction import move_images_to_destination, organize_images_by_type
from modules.preprocessing import (
    preprocess_csv,
    merge_csv_with_cloud_data,
    add_image_paths_to_csv,
    clean_final_dataset,
    create_initial_csv_files,
)
from modules.data_creation import create_and_upload_dataset
from modules.embedding_generation import (
    generate_embeddings_from_dataset,
    get_embedding_size,
)


def setup_directories():
    """
    Set up the necessary directories for the pipeline.
    """
    ensure_dir(config.IMAGE_ZIPS_DIR)
    ensure_dir(config.RAW_IMAGES_DIR)
    ensure_dir(config.PROCESSED_IMAGES_DIR)
    ensure_dir(config.CSV_DIR)
    ensure_dir(config.EMBEDDINGS_DIR)


def download_phase(start_date, end_date):
    """
    Execute the image download phase.

    Args:
        start_date (datetime.date): Start date for downloading.
        end_date (datetime.date): End date for downloading.
    """
    print(f"\n--- PHASE 1: Downloading images from {start_date} to {end_date} ---")
    download_images_in_range(
        start_date=start_date,
        end_date=end_date,
        base_save_path=config.IMAGE_ZIPS_DIR,
        base_url=config.BASE_URL,
    )


def extraction_phase():
    """
    Execute the image extraction and organization phase.
    """
    print("\n--- PHASE 2: Extracting and organizing images ---")

    # First, move all images from date folders to a central location
    print("Moving images from date folders to central location...")
    move_images_to_destination(
        src_directory=config.IMAGE_ZIPS_DIR,
        dest_directory=os.path.join(config.ROOT_DIR, "images"),
    )

    # Then, organize images into raw and processed folders
    print("Organizing images by type...")
    organize_images_by_type(
        source_directory=os.path.join(config.ROOT_DIR, "images"),
        raw_dir=config.RAW_IMAGES_DIR,
        processed_dir=config.PROCESSED_IMAGES_DIR,
        raw_pattern=config.RAW_IMAGE_PATTERN,
        processed_pattern=config.PROCESSED_IMAGE_PATTERN,
    )


def preprocessing_phase(main_csv, cloud_csv, output_csv, start_date, end_date):
    """
    Execute the data preprocessing phase.

    Args:
        main_csv (str): Path to main weather data CSV.
        cloud_csv (str): Path to cloud data CSV.
        output_csv (str): Path to output final CSV.
        start_date (datetime.date): Start date for data.
        end_date (datetime.date): End date for data.
    """
    print("\n--- PHASE 3: Preprocessing data ---")

    # Check if CSV files exist, create if they don't
    print("Checking for CSV files...")
    if not os.path.exists(main_csv) or not os.path.exists(cloud_csv):
        print("CSV files not found. Creating initial files from images...")
        create_initial_csv_files(
            main_csv_path=main_csv,
            cloud_csv_path=cloud_csv,
            raw_images_dir=config.RAW_IMAGES_DIR,
            processed_images_dir=config.PROCESSED_IMAGES_DIR,
            start_date=start_date,
            end_date=end_date,
        )

    # Preprocess main CSV
    print("Preprocessing main CSV...")
    main_df = preprocess_csv(main_csv)

    # Merge with cloud data
    print("Merging with cloud data...")
    merged_df = merge_csv_with_cloud_data(
        main_csv_path=main_csv,
        cloud_data_csv_path=cloud_csv,
        output_path=os.path.join(config.CSV_DIR, "merged_data.csv"),
    )

    # Add image paths
    print("Adding image paths...")
    paths_df = add_image_paths_to_csv(
        csv_path=os.path.join(config.CSV_DIR, "merged_data.csv"),
        raw_images_dir=config.RAW_IMAGES_DIR,
        processed_images_dir=config.PROCESSED_IMAGES_DIR,
        output_path=os.path.join(config.CSV_DIR, "data_with_paths.csv"),
    )

    # Clean final dataset
    print("Creating final clean dataset...")
    final_df = clean_final_dataset(
        csv_path=os.path.join(config.CSV_DIR, "data_with_paths.csv"),
        output_path=output_csv,
        required_columns=["Raw_images", "Processed_Images"],
    )

    print(f"Final dataset saved to {output_csv} with {len(final_df)} rows")


def dataset_creation_phase(final_csv, repo_name, token=None):
    """
    Execute the dataset creation and upload phase.

    Args:
        final_csv (str): Path to final CSV.
        repo_name (str): HF repo name.
        token (str, optional): HF token.
    """
    print("\n--- PHASE 4: Creating and uploading dataset ---")

    # If token is not provided, check environment variable
    if token is None:
        token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")
        if token:
            print("Using Hugging Face token from environment variable")
        else:
            print("No Hugging Face token found in environment variables")

    success = create_and_upload_dataset(
        csv_path=final_csv,
        repo_name=repo_name,
        token=token,
        columns_to_include=config.DATASET_COLUMNS,
    )

    if success:
        print(f"Dataset successfully created and uploaded to {repo_name}")
    else:
        print("Dataset creation or upload failed")


def embedding_generation_phase(
    repo_name,
    output_file=None,
    model_name=None,
    batch_size=1,
    width_padding=0,
    height_padding=0,
):
    """
    Execute the image embedding generation phase.

    Args:
        repo_name (str): HF repo name to generate embeddings from.
        output_file (str, optional): Path to save embeddings.
        model_name (str, optional): Model to use for generating embeddings.
        batch_size (int): Batch size for processing.
        width_padding (int): Padding to add to image width.
        height_padding (int): Padding to add to image height.

    Returns:
        str: Path to the generated embeddings file.
    """
    print("\n--- PHASE 5: Generating image embeddings ---")

    # Set default model if not provided
    if model_name is None:
        model_name = config.DEFAULT_MODEL_NAME
        print(f"Using default model: {model_name}")

    # Create a default output file name if not provided
    if output_file is None:
        model_short_name = model_name.split("/")[-1].replace("-", "_").lower()
        timestamp = datetime.datetime.now().strftime("%Y%m%d")
        output_file = os.path.join(
            config.EMBEDDINGS_DIR, f"{model_short_name}_{timestamp}.json"
        )

    print(f"Generating embeddings using {model_name} model")
    print(f"Output will be saved to: {output_file}")

    # Generate embeddings
    count = generate_embeddings_from_dataset(
        dataset_name=repo_name,
        output_file=output_file,
        model_name=model_name,
        batch_size=batch_size,
        padding=(width_padding, height_padding),
    )

    # Get embedding size
    embed_size = get_embedding_size(output_file)
    if embed_size:
        print(f"Generated {count} embeddings with dimensionality of {embed_size}")

    return output_file


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Data processing pipeline for solar image dataset"
    )

    # Data acquisition and processing arguments
    parser.add_argument(
        "--start-date",
        type=str,
        default="2020-10-26",
        help="Start date for downloading images (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--end-date",
        type=str,
        default="2020-12-31",
        help="End date for downloading images (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--main-csv",
        type=str,
        default=os.path.join(config.CSV_DIR, "20200101.csv"),
        help="Path to main CSV file",
    )
    parser.add_argument(
        "--cloud-csv",
        type=str,
        default=os.path.join(config.CSV_DIR, "add.csv"),
        help="Path to cloud data CSV file",
    )
    parser.add_argument(
        "--output-csv",
        type=str,
        default=os.path.join(config.CSV_DIR, "2020_data_final.csv"),
        help="Path to output final CSV file",
    )

    # Hugging Face arguments
    parser.add_argument(
        "--repo-name",
        type=str,
        default=config.HF_REPO,
        help="HuggingFace repository name (username/dataset-name)",
    )
    parser.add_argument(
        "--token",
        type=str,
        default=None,
        help="HuggingFace API token (if not provided, will look for HF_TOKEN or HUGGINGFACE_TOKEN environment variable)",
    )

    # Embedding generation arguments
    parser.add_argument(
        "--embedding-model",
        type=str,
        default=config.DEFAULT_MODEL_NAME,
        help="Model to use for generating embeddings",
    )
    parser.add_argument(
        "--embedding-output",
        type=str,
        default=None,
        help="Path to save embeddings (if not provided, will generate a default name)",
    )
    parser.add_argument(
        "--batch-size", type=int, default=1, help="Batch size for embedding generation"
    )
    parser.add_argument(
        "--width-padding",
        type=int,
        default=config.IMAGE_PADDING[0],
        help="Padding to add to image width (for embedding generation)",
    )
    parser.add_argument(
        "--height-padding",
        type=int,
        default=config.IMAGE_PADDING[1],
        help="Padding to add to image height (for embedding generation)",
    )

    # Phase control arguments
    parser.add_argument(
        "--skip-download", action="store_true", help="Skip the download phase"
    )
    parser.add_argument(
        "--skip-extraction", action="store_true", help="Skip the extraction phase"
    )
    parser.add_argument(
        "--skip-preprocessing", action="store_true", help="Skip the preprocessing phase"
    )
    parser.add_argument(
        "--skip-dataset-creation",
        action="store_true",
        help="Skip the dataset creation phase",
    )
    parser.add_argument(
        "--skip-embedding-generation",
        action="store_true",
        help="Skip the embedding generation phase",
    )
    parser.add_argument(
        "--only-embedding-generation",
        action="store_true",
        help="Only run the embedding generation phase",
    )

    return parser.parse_args()


def main():
    """Execute the full data processing pipeline."""
    args = parse_args()

    # Setup directories
    setup_directories()

    # Parse dates
    start_date = datetime.datetime.strptime(args.start_date, "%Y-%m-%d").date()
    end_date = datetime.datetime.strptime(args.end_date, "%Y-%m-%d").date()

    # If only embedding generation flag is set, skip to that phase
    if args.only_embedding_generation:
        print("Running only embedding generation phase...")
        embedding_generation_phase(
            args.repo_name,
            args.embedding_output,
            args.embedding_model,
            args.batch_size,
            args.width_padding,
            args.height_padding,
        )
        print("\nEmbedding generation completed successfully!")
        return

    # Execute phases
    if not args.skip_download:
        download_phase(start_date, end_date)
    else:
        print("Skipping download phase...")

    if not args.skip_extraction:
        extraction_phase()
    else:
        print("Skipping extraction phase...")

    if not args.skip_preprocessing:
        preprocessing_phase(
            args.main_csv, args.cloud_csv, args.output_csv, start_date, end_date
        )
    else:
        print("Skipping preprocessing phase...")

    if not args.skip_dataset_creation:
        dataset_creation_phase(args.output_csv, args.repo_name, args.token)
    else:
        print("Skipping dataset creation phase...")

    if not args.skip_embedding_generation:
        embedding_generation_phase(
            args.repo_name,
            args.embedding_output,
            args.embedding_model,
            args.batch_size,
            args.width_padding,
            args.height_padding,
        )
    else:
        print("Skipping embedding generation phase...")

    print("\nPipeline completed successfully!")


if __name__ == "__main__":
    main()
