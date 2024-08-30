import torch
from transformers import ViTModel, ViTFeatureExtractor
from PIL import Image
import numpy as np
import os
from tqdm import tqdm

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
        features = outputs.last_hidden_state.mean(dim=1).squeeze().numpy()
    
    return features

# Directory where your dataset images are stored
dataset_dir = 'dataset'

features = []
image_paths = []

for img_name in tqdm(os.listdir(dataset_dir)):
    img_path = os.path.join(dataset_dir, img_name)
    image_paths.append(img_path)
    feature = extract_features_vit(img_path)
    features.append(feature)

features = np.array(features)
image_paths = np.array(image_paths)

np.savez('pre_extracted_features.npz', features=features, image_paths=image_paths)
print(f"Saved features for {len(image_paths)} images.")