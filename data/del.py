"""
Module for generating image embeddings from the processed dataset.
"""
import os
import torch
import json
from tqdm import tqdm
from pathlib import Path
from PIL import Image
from transformers import AutoImageProcessor, AutoModel
from datasets import load_dataset

def add_padding_to_image(image, width_padding, height_padding):
    """
    Add padding to an image to meet model input requirements.
    
    Args:
        image (PIL.Image): Input image.
        width_padding (int): Padding to add on left and right sides.
        height_padding (int): Padding to add on top and bottom sides.
        
    Returns:
        PIL.Image: Padded image.
    """
    width, height = image.size
    new_width = width + 2 * width_padding
    new_height = height + 2 * height_padding
    new_image = Image.new("RGB", (new_width, new_height), color="black")
    new_image.paste(image, (width_padding, height_padding))
    return new_image

def initialize_model(model_name, device=None):
    """
    Initialize image processor and model.
    
    Args:
        model_name (str): Name or path of the pre-trained model.
        device (str, optional): Device to use for computation.
        
    Returns:
        tuple: (processor, model, device) tuple.
    """
    # Set up the environment for better memory management
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
    
    # Determine device
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load processor and model
    processor = AutoImageProcessor.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModel.from_pretrained(model_name, trust_remote_code=True)
    model = model.to(device)
    model.eval()  # Set model to evaluation mode
    
    return processor, model, device

def generate_embeddings(images, processor, model, device):
    """
    Generate embeddings for a batch of images.
    
    Args:
        images (list): List of PIL image objects.
        processor: Image processor.
        model: Model for generating embeddings.
        device: Device to use for computation.
        
    Returns:
        torch.Tensor: Batch of embeddings.
    """
    with torch.no_grad():
        inputs = processor(images=images, return_tensors="pt").to(device)
        outputs = model(**inputs)
        last_hidden_states = outputs.last_hidden_state
        cls_embeddings = last_hidden_states[:, 0, :]
    return cls_embeddings

def generate_embeddings_from_dataset(
    dataset_name,
    output_file,
    model_name="facebook/dinov2-giant",
    batch_size=1,
    padding=(0, 0),
    streaming=True,
    split="train"
):
    """
    Generate embeddings from a HuggingFace dataset and save to JSONL file.
    
    Args:
        dataset_name (str): Name of the HuggingFace dataset.
        output_file (str): Path to save embeddings.
        model_name (str): Name of the pre-trained model to use.
        batch_size (int): Batch size for processing.
        padding (tuple): (width_padding, height_padding) to apply to images.
        streaming (bool): Whether to stream the dataset.
        split (str): Dataset split to use.
        
    Returns:
        int: Number of embeddings generated.
    """
    # Initialize model
    processor, model, device = initialize_model(model_name)
    
    # Ensure output directory exists
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Clear the output file
    with open(output_file, "w") as f:
        pass
    
    # Load the dataset
    print(f"Loading dataset {dataset_name}...")
    dataset = load_dataset(dataset_name, split=split, streaming=streaming)
    
    # Process the dataset
    count = 0
    batched_images = []
    batched_indices = []
    
    for idx, row in tqdm(enumerate(dataset), desc=f"Generating embeddings for {dataset_name}"):
        try:
            # Get the image
            image = row['Raw_images']
            
            # Apply padding if needed
            if padding[0] > 0 or padding[1] > 0:
                image = add_padding_to_image(image, padding[0], padding[1])
            
            batched_images.append(image)
            batched_indices.append(idx)
            
            # Process batch when it reaches the specified size
            if len(batched_images) == batch_size:
                embeddings = generate_embeddings(batched_images, processor, model, device)
                
                # Save embeddings
                for i, embedding in enumerate(embeddings):
                    entry = {
                        "index": batched_indices[i],
                        "embedding": embedding.cpu().squeeze().tolist()
                    }
                    
                    with open(output_file, "a") as f:
                        json.dump(entry, f)
                        f.write("\n")
                
                count += len(batched_images)
                batched_images = []
                batched_indices = []
        
        except Exception as e:
            print(f"Error processing image at index {idx}: {e}")
    
    # Process any remaining images
    if batched_images:
        embeddings = generate_embeddings(batched_images, processor, model, device)
        for i, embedding in enumerate(embeddings):
            entry = {
                "index": batched_indices[i],
                "embedding": embedding.cpu().squeeze().tolist()
            }
            with open(output_file, "a") as f:
                json.dump(entry, f)
                f.write("\n")
        
        count += len(batched_images)
    
    print(f"Generated {count} embeddings saved to {output_file}")
    return count

def get_embedding_size(jsonl_file):
    """
    Get the size (dimensionality) of embeddings in a JSONL file.
    
    Args:
        jsonl_file (str): Path to the JSONL file.
        
    Returns:
        int: Size of the embedding vector.
    """
    try:
        with open(jsonl_file, 'r') as file:
            # Read the first line
            first_line = file.readline().strip()
            # Parse the JSON object
            data = json.loads(first_line)
            # Check if the "embedding" field exists
            if "embedding" in data and isinstance(data["embedding"], list):
                return len(data["embedding"])
            else:
                print("The first line does not contain a valid 'embedding' field.")
                return None
    except Exception as e:
        print(f"Error reading embedding size: {e}")
        return None