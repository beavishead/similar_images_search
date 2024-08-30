import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import logging
from PIL import Image
import torch
from transformers import ViTModel, ViTFeatureExtractor
import os

logging.basicConfig(level=logging.INFO)

# Load pre-extracted features
pre_extracted_data = np.load('pre_extracted_features.npz')
dataset_features = pre_extracted_data['features']
image_paths = pre_extracted_data['image_paths']
logging.info(f"Loaded pre-extracted features for {len(image_paths)} images.")

# Load the model and feature extractor
model = ViTModel.from_pretrained('google/vit-base-patch16-224')
feature_extractor = ViTFeatureExtractor.from_pretrained('google/vit-base-patch16-224')

# Load your custom weights
state_dict = torch.load('model_weights.pth', map_location=torch.device('cpu'))
new_state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
model.load_state_dict(new_state_dict, strict=False)
print("Loaded pre-trained weights successfully.")
model.eval()


def extract_features_vit(img_path):
    img = Image.open(img_path).convert('RGB')
    inputs = feature_extractor(images=img, return_tensors="pt")
    
    with torch.no_grad():
        outputs = model(**inputs)
        # Use the [CLS] token output as the image representation
        features = outputs.last_hidden_state[:, 0, :].squeeze().numpy()
    
    return features

def get_similar_images(uploaded_image_path):
    # Extract features for the uploaded image
    uploaded_feature = extract_features_vit(uploaded_image_path)
    
    # Ensure the uploaded feature is 2D for cosine_similarity
    uploaded_feature = uploaded_feature.reshape(1, -1)
    
    # Calculate cosine similarity
    similarity_scores = cosine_similarity(uploaded_feature, dataset_features)[0]
    
    # Get indices of top 6 similar images
    top_indices = np.argsort(similarity_scores)[-6:][::-1]
    
    # Create list of similar images with their paths and scores
    similar_images = [
        {
            'path': os.path.basename(image_paths[i]),  # Use basename to get just the filename
            'similarity': float(similarity_scores[i])  # Convert to float for JSON serialization
        }
        for i in top_indices
    ]
    
    return similar_images