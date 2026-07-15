import os
import glob
import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel, pipeline
from qdrant_client import QdrantClient
from qdrant_client.http import models as qdrant_models
from tqdm import tqdm

def main():
    print("Initializing Qdrant Client...")
    client = QdrantClient(path="data/qdrant_storage")
    
    collection_name = "fashion_collection"
    
    print(f"Creating/Recreating collection '{collection_name}'...")
    client.recreate_collection(
        collection_name=collection_name,
        vectors_config=qdrant_models.VectorParams(
            size=512,
            distance=qdrant_models.Distance.COSINE
        )
    )
    
    print("Loading CLIP model and processor...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    model_id = "openai/clip-vit-base-patch32"
    model = CLIPModel.from_pretrained(model_id).to(device)
    processor = CLIPProcessor.from_pretrained(model_id)
    
    print("Loading YOLOS object detector for person cropping...")
    # YOLOS is very lightweight and runs quickly on CPU
    detector = pipeline("object-detection", model="hustvl/yolos-tiny", device=-1 if device == "cpu" else 0)
    
    # Define our attribute lexicons
    colors = ["black", "white", "blue", "red", "yellow", "green", "grey", "brown", "pink", "purple", "orange", "beige"]
    upper_types = ["shirt", "t-shirt", "hoodie", "blazer", "raincoat", "jacket", "dress", "sweater"]
    lower_types = ["pants", "jeans", "shorts", "skirt"]
    environments = ["office", "urban street", "park", "home", "beach", "nature path", "formal hall"]
    styles = ["formal business attire", "casual weekend wear", "sporty activewear", "cozy loungewear", "trendy streetwear"]
    
    # Precompute text features for fast zero-shot classification
    def precompute_text_features(labels, prompt_template):
        prompts = [prompt_template.format(label) for label in labels]
        inputs = processor(text=prompts, return_tensors="pt", padding=True).to(device)
        with torch.no_grad():
            features = model.get_text_features(**inputs)
            if hasattr(features, "pooler_output"):
                features = features.pooler_output
        features = features / features.norm(dim=-1, keepdim=True)
        return features

    print("Precomputing text embeddings for attributes...")
    color_features = precompute_text_features(colors, "a photo of a person wearing a {} clothing item")
    upper_features = precompute_text_features(upper_types, "a photo of a person wearing a {}")
    lower_features = precompute_text_features(lower_types, "a photo of a person wearing {}")
    env_features = precompute_text_features(environments, "a photo of a person in a {} setting")
    style_features = precompute_text_features(styles, "a photo of someone dressed in {}")
    tie_color_features = precompute_text_features(colors, "a photo of a person wearing a {} tie")
    
    # Fetch all images
    image_paths = glob.glob("data/images/*.jpg")
    print(f"Found {len(image_paths)} images to index.")
    
    points = []
    
    for idx, img_path in enumerate(tqdm(image_paths, desc="Indexing images")):
        try:
            # Load image
            img = Image.open(img_path).convert("RGB")
            W, H = img.size
            
            # Detect objects (person, tie, etc.)
            detections = detector(img)
            persons = [d for d in detections if d['label'] == 'person' and d['score'] > 0.45]
            
            person_found = False
            upper_crop = None
            lower_crop = None
            
            if persons:
                # Find the largest person bounding box by area
                def get_area(p):
                    b = p['box']
                    return (b['xmax'] - b['xmin']) * (b['ymax'] - b['ymin'])
                
                best_person = max(persons, key=get_area)
                b = best_person['box']
                
                # Clip box to image boundaries
                xmin = max(0, int(b['xmin']))
                ymin = max(0, int(b['ymin']))
                xmax = min(W, int(b['xmax']))
                ymax = min(H, int(b['ymax']))
                
                if (xmax - xmin) > 30 and (ymax - ymin) > 30:
                    person_found = True
                    # Crop only the person area
                    person_crop = img.crop((xmin, ymin, xmax, ymax))
                    p_w, p_h = person_crop.size
                    
                    # Split the person's body into upper and lower parts
                    upper_crop = person_crop.crop((0, 0, p_w, int(p_h * 0.55)))
                    lower_crop = person_crop.crop((0, int(p_h * 0.45), p_w, p_h))
            
            # Fallback if no person is detected
            if not person_found:
                upper_crop = img.crop((0, 0, W, int(H * 0.55)))
                lower_crop = img.crop((0, int(H * 0.45), W, H))
            
            # Helper function to predict attributes on crops
            def predict_label(crop_img, precomputed_features, labels):
                inputs = processor(images=crop_img, return_tensors="pt").to(device)
                with torch.no_grad():
                    img_features = model.get_image_features(**inputs)
                    if hasattr(img_features, "pooler_output"):
                        img_features = img_features.pooler_output
                img_features = img_features / img_features.norm(dim=-1, keepdim=True)
                similarities = (img_features @ precomputed_features.T).squeeze(0)
                probs = (similarities * 100).softmax(dim=-1)
                best_idx = probs.argmax().item()
                return labels[best_idx], probs[best_idx].item()
            
            # Predict garment types and colors
            upper_color, _ = predict_label(upper_crop, color_features, colors)
            upper_type, _ = predict_label(upper_crop, upper_features, upper_types)
            
            lower_color, _ = predict_label(lower_crop, color_features, colors)
            lower_type, _ = predict_label(lower_crop, lower_features, lower_types)
            
            env, _ = predict_label(img, env_features, environments)
            style, _ = predict_label(img, style_features, styles)
            
            # Use YOLOS for highly precise tie detection (checks COCO tie label)
            has_tie = any(d['label'] == 'tie' and d['score'] > 0.22 for d in detections)
            
            if has_tie:
                # Classify tie color on the person's upper crop
                tie_color, _ = predict_label(upper_crop, tie_color_features, colors)
            else:
                tie_color = "none"
                
            # Extract global image embedding
            inputs_full = processor(images=img, return_tensors="pt").to(device)
            with torch.no_grad():
                global_img_features = model.get_image_features(**inputs_full)
                if hasattr(global_img_features, "pooler_output"):
                    global_img_features = global_img_features.pooler_output
            global_img_features = global_img_features / global_img_features.norm(dim=-1, keepdim=True)
            global_vector = global_img_features.squeeze(0).cpu().numpy().tolist()
            
            # Save payload
            payload = {
                "filename": os.path.basename(img_path),
                "path": f"data/images/{os.path.basename(img_path)}",
                "upper_color": upper_color,
                "upper_type": upper_type,
                "lower_color": lower_color,
                "lower_type": lower_type,
                "has_tie": has_tie,
                "tie_color": tie_color,
                "environment": env,
                "style": style
            }
            
            points.append(
                qdrant_models.PointStruct(
                    id=idx,
                    vector=global_vector,
                    payload=payload
                )
            )
            
        except Exception as e:
            print(f"Error processing image {img_path}: {e}")
            
    print(f"Upserting {len(points)} vectors to Qdrant...")
    chunk_size = 100
    for i in range(0, len(points), chunk_size):
        chunk = points[i:i+chunk_size]
        client.upsert(
            collection_name=collection_name,
            points=chunk
        )
        
    print("Indexing completed successfully!")
    print(f"Indexed {len(points)} items in collection '{collection_name}'.")

if __name__ == "__main__":
    main()
