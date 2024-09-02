import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import logging
from PIL import Image
import torch
from transformers import ViTModel, ViTFeatureExtractor
import os
import gdown

logging.basicConfig(level=logging.INFO)

def download_file(file_id, output_path):
    if not os.path.exists(output_path):
        url = f"https://drive.google.com/uc?export=download&id={file_id}"
        gdown.download(url, output_path, quiet=False)
        print(f"Downloaded {output_path}")
    else:
        print(f"{output_path} already exists.")

# Load the model and feature extractor
model = ViTModel.from_pretrained('google/vit-base-patch16-224')
feature_extractor = ViTFeatureExtractor.from_pretrained('google/vit-base-patch16-224')

def load_model(weights_path):
    global model
    state_dict = torch.load(weights_path, map_location=torch.device('cpu'))
    new_state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    model.load_state_dict(new_state_dict, strict=False)
    print("Loaded pre-trained weights successfully.")
    model.eval()

def load_features(features_path):
    pre_extracted_data = np.load(features_path)
    dataset_features = pre_extracted_data['features']
    image_paths = pre_extracted_data['image_paths']
    logging.info(f"Loaded pre-extracted features for {len(image_paths)} images.")
    return dataset_features, image_paths

def extract_features_vit(img_path):
    img = Image.open(img_path).convert('RGB')
    inputs = feature_extractor(images=img, return_tensors="pt")
    
    with torch.no_grad():
        outputs = model(**inputs)
        features = outputs.last_hidden_state[:, 0, :].squeeze().numpy()
    
    return features

def get_similar_images(uploaded_image_path, dataset_features, image_paths):
    uploaded_feature = extract_features_vit(uploaded_image_path)
    uploaded_feature = uploaded_feature.reshape(1, -1)
    similarity_scores = cosine_similarity(uploaded_feature, dataset_features)[0]
    top_indices = np.argsort(similarity_scores)[-6:][::-1]
    similar_images = [
        {
            'path': os.path.basename(image_paths[i]),
            'similarity': float(similarity_scores[i])
        }
        for i in top_indices
    ]
    return similar_images