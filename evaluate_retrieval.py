import os
import json
import torch
import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer
from PIL import Image

# repo helpers
from llava.model.builder import load_pretrained_model
from llava.mm_utils import (
    tokenizer_image_token,
    process_highres_image,
    process_anyres_image,
    process_highres_image_crop_split,
)
from llava.constants import IMAGE_TOKEN_INDEX

def calculate_metrics(sim_matrix, text_to_image_idx):
    """
    Calculates metrics by finding the rank of the target unique image.
    sim_matrix shape: (num_texts, num_unique_images)
    text_to_image_idx: list mapping text index to the correct unique image index
    """
    ranks = np.zeros(sim_matrix.shape[0])
    for i in range(sim_matrix.shape[0]):
        # Sort image indices by descending similarity
        d_i = np.argsort(sim_matrix[i])[::-1]
        
        # Find where the correct unique image is in the sorted list
        target_idx = text_to_image_idx[i]
        ranks[i] = np.where(d_i == target_idx)[0][0]

    r1 = 100.0 * len(np.where(ranks < 1)[0]) / len(ranks)
    r5 = 100.0 * len(np.where(ranks < 5)[0]) / len(ranks)
    r10 = 100.0 * len(np.where(ranks < 10)[0]) / len(ranks)
    medr = np.floor(np.median(ranks)) + 1
    
    return {"Recall@1": r1, "Recall@5": r5, "Recall@10": r10, "MedR": medr}

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True, help="Path to your trained checkpoint")
    parser.add_argument("--data_path", type=str, required=True, help="Path to Flickr30k JSON")
    parser.add_argument("--image_folder", type=str, required=True, help="Path to Flickr30k images")
    args = parser.parse_args()

    print(f"Loading Model from {args.model_path}...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model, tokenizer, and image processor using repo helper
    tokenizer, model, image_processor, _ = load_pretrained_model(args.model_path, None, "llava_qwen")
    model.to(device)
    model.eval()

    # Load data
    print(f"Loading data from {args.data_path}...")
    with open(args.data_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    # take subset for evaluation
    test_data = data[-5000:]

    all_text_embeds = []
    unique_image_embeds = []
    unique_image_paths = []
    text_to_image_idx = [] # Maps each text to its corresponding unique image

    def process_image_local(image_file, image_folder, image_processor, image_aspect_ratio="square", image_grid_pinpoints=None, overwrite_image_aspect_ratio=None):
        if image_file is None or image_file == "":
            crop_size = image_processor.crop_size
            return torch.zeros(1, 3, crop_size["height"], crop_size["width"]), (crop_size["width"], crop_size["height"]), "text"

        try:
            image = Image.open(os.path.join(image_folder, image_file)).convert("RGB")
        except Exception as exn:
            print(f"Failed to open image {image_file}. Exception:", exn)
            raise exn

        image_size = image.size
        ia = image_aspect_ratio
        if overwrite_image_aspect_ratio is not None:
            ia = overwrite_image_aspect_ratio
        if ia == "highres":
            image = process_highres_image(image, image_processor, image_grid_pinpoints)
        elif ia == "anyres" or "anyres_max" in ia:
            image = process_anyres_image(image, image_processor, image_grid_pinpoints)
        elif ia == "crop_split":
            image = process_highres_image_crop_split(image, dict(image_processor=image_processor, image_grid_pinpoints=image_grid_pinpoints))
        elif ia == "pad":
            def expand2square(pil_img, background_color):
                width, height = pil_img.size
                if width == height: return pil_img
                elif width > height:
                    result = Image.new(pil_img.mode, (width, width), background_color)
                    result.paste(pil_img, (0, (width - height) // 2))
                    return result
                else:
                    result = Image.new(pil_img.mode, (height, height), background_color)
                    result.paste(pil_img, ((height - width) // 2, 0))
                    return result

            image = expand2square(image, tuple(int(x * 255) for x in image_processor.image_mean))
            image = image_processor.preprocess(image, return_tensors="pt")["pixel_values"][0]
        else:
            image = image_processor.preprocess(image, return_tensors="pt")["pixel_values"][0]
        return image, image_size, "image"

    print("Extracting Embeddings...")
    with torch.no_grad():
        for item in tqdm(test_data):
            # --- TEXT EXTRACTION ---
            conversations = item.get("conversations", [])
            text_query = ""
            if isinstance(conversations, list) and len(conversations) >= 2:
                second = conversations[1]
                if isinstance(second, dict):
                    text_query = second.get("value", "")

            if not text_query:
                text_query = (item.get("qry") or item.get("query") or item.get("caption") or 
                              item.get("text") or item.get("sentence") or item.get("raw"))

            text_query = str(text_query).strip() or "a photo"

            input_ids = tokenizer_image_token(text_query, tokenizer, return_tensors="pt")
            if input_ids.ndim == 1:
                input_ids = input_ids.unsqueeze(0)
            qry_inputs = {"input_ids": input_ids.to(device)}

            # use float32 for stability
            with torch.cuda.amp.autocast(enabled=False):
                text_embed = model.encode_multimodal_embeddings(**qry_inputs, pooling_type="avg")
            text_embed = text_embed.float()
            # Replace any NaNs/Infs that can arise from mixed precision ops
            text_embed = torch.nan_to_num(text_embed, nan=0.0, posinf=1e3, neginf=-1e3)
            text_embed = torch.nn.functional.normalize(text_embed, p=2, dim=-1)
            all_text_embeds.append(text_embed.cpu())

            # --- IMAGE EXTRACTION (UNIQUE ONLY) ---
            pos_image_path = item.get("pos_image_path") or item.get("qry_image_path") or (item.get("image") and os.path.basename(item.get("image")))
            
            # Only process the image if we haven't seen it yet
            if pos_image_path not in unique_image_paths:
                unique_image_paths.append(pos_image_path)
                image_tensor, image_size, _ = process_image_local(pos_image_path, args.image_folder, image_processor)
                image_tensor = image_tensor.to(device=device, dtype=model.dtype)
                
                # include assistant token in prompt
                prompt = "USER: <image>\nDescribe this image. ASSISTANT:"
                img_input_ids = tokenizer_image_token(prompt, tokenizer, return_tensors="pt")
                if img_input_ids.ndim == 1:
                    img_input_ids = img_input_ids.unsqueeze(0)
                    
                pos_inputs = {
                    "input_ids": img_input_ids.to(device),
                    "images": image_tensor.unsqueeze(0),
                    "image_sizes": [[int(image_size[0]), int(image_size[1])]],
                    "modalities": ["image"],
                }
                # use float32 for stability
                with torch.cuda.amp.autocast(enabled=False):
                    image_embed = model.encode_multimodal_embeddings(**pos_inputs, pooling_type="avg")
                image_embed = image_embed.float()
                image_embed = torch.nan_to_num(image_embed, nan=0.0, posinf=1e3, neginf=-1e3)
                image_embed = torch.nn.functional.normalize(image_embed, p=2, dim=-1)
                unique_image_embeds.append(image_embed.cpu())
            
            # Map this text to the correct unique image index
            target_idx = unique_image_paths.index(pos_image_path)
            text_to_image_idx.append(target_idx)

    # Stack embeddings
    text_features = torch.cat(all_text_embeds, dim=0).float()      # Shape: (1000, Dim)
    image_features = torch.cat(unique_image_embeds, dim=0).float() # Shape: (200, Dim)

    # Compute Similarity Matrix
    print(f"Computing Similarity Matrix... (Texts: {text_features.shape[0]}, Unique Images: {image_features.shape[0]})")
    sim_matrix = (text_features @ image_features.T).numpy()
    print(f"Similarity Range: Min={sim_matrix.min():.4f}, Max={sim_matrix.max():.4f}, Mean={sim_matrix.mean():.4f}")

    metrics = calculate_metrics(sim_matrix, text_to_image_idx)
    print("\n--- Final Evaluation Metrics ---")
    print(f"Recall@1:  {metrics['Recall@1']:.2f}")
    print(f"Recall@5:  {metrics['Recall@5']:.2f}")
    print(f"Recall@10: {metrics['Recall@10']:.2f}")
    print(f"MedR:      {int(metrics['MedR'])}")

if __name__ == "__main__":
    main()