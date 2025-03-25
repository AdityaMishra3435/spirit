## Usage

The data processing pipeline is divided into five main phases:

1. **Download**: Download image zip files from the NREL website.
2. **Extraction**: Extract and organize images from the downloaded zip files.
3. **Preprocessing**: Process CSV data and merge with image information.
4. **Dataset Creation**: Create and upload a Hugging Face dataset.
5. **Embedding Generation**: Generate image embeddings using pre-trained models.

To run the full pipeline:

```
python main.py --start-date 2020-10-26 --end-date 2020-12-31
```

### Command Line Arguments

#### Data Acquisition and Processing
- `--start-date`: Start date for image downloading (YYYY-MM-DD)
- `--end-date`: End date for image downloading (YYYY-MM-DD)
- `--main-csv`: Path to main weather data CSV
- `--cloud-csv`: Path to cloud data CSV
- `--output-csv`: Path for final CSV output

#### Hugging Face Integration
- `--repo-name`: HuggingFace repository name (username/dataset-name)
- `--token`: HuggingFace API token (if not provided, will look for environment variables)

#### Embedding Generation
- `--embedding-model`: Model to use for generating embeddings (default: facebook/dinov2-giant)
- `--embedding-output`: Path to save embeddings JSON file
- `--batch-size`: Batch size for embedding generation
- `--width-padding`: Padding to add to image width (for models requiring specific dimensions)
- `--height-padding`: Padding to add to image height (for models requiring specific dimensions)

#### Phase Control
- `--skip-download`: Skip the download phase
- `--skip-extraction`: Skip the extraction phase
- `--skip-preprocessing`: Skip the preprocessing phase
- `--skip-dataset-creation`: Skip the dataset creation phase
- `--skip-embedding-generation`: Skip the embedding generation phase
- `--only-embedding-generation`: Only run the embedding generation phase

### Running Only the Embedding Generation

If you already have the dataset uploaded to Hugging Face and want to generate embeddings:

```bash
python main.py --only-embedding-generation --repo-name "your-username/dataset-name" --embedding-model "google/vit-base-patch16-224"
```

## Data Structure

The project generates and processes the following data:

### Directories
- `image-zips/`: Downloaded zip files containing images
- `images/`: Extracted images from all dates
- `raw_images/`: Organized raw image files with standardized names
- `processed_images/`: Organized processed image files with standardized names
- `csv/`: CSV files containing meteorological data and image metadata
- `embeddings/`: Generated image embeddings in JSON format

### Dataset Files
- The final dataset includes:
  - Meteorological data (irradiance measurements)
  - Cloud cover information
  - Paths to raw and processed images

### Embeddings Files
- Embeddings are stored in JSON Line (JSONL) format:
  - Each line contains a JSON object with:
    - `"index"`: The row index in the dataset
    - `"embedding"`: The vector representation of the image

### Models Supported
The embedding generation supports various vision models from Hugging Face, including:
- `google/vit-base-patch16-224` (default)
- `facebook/dinov2-giant` 
- `facebook/dinov2-base`
- `apple/aimv2-3B-patch14-448`  
And other vision models compatible with the Transformers library