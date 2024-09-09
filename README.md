# Image Similarity Search

An AI-powered application that finds similar images using deep learning techniques.

## Project Overview

This project implements an image similarity search using a pre-trained Vision Transformer (ViT) model. It allows users to upload an image and find visually similar images from a pre-defined dataset.

## Examples of the search with UI

![image]("static/for _readme/example_with_roses.jpg")
![image]("static/for _readme/example_with_taj_mahal.jpg")

## Key Features

- Utilizes a Vision Transformer (ViT) model for feature extraction [Hugging Face model page](https://img.shields.io/badge/🤗%20Model-ViT--Base--Patch16--224-yellow)(https://huggingface.co/google/vit-base-patch16-224)
- Pre-extracted features for quick similarity comparisons
- Flask web application for easy interaction
- Supports various image formats (PNG, JPG, JPEG, GIF)

## Technologies Used

- ![Python](https://img.shields.io/badge/-Python-3776AB?style=flat-square&logo=Python&logoColor=white)
- ![PyTorch](https://img.shields.io/badge/-PyTorch-EE4C2C?style=flat-square&logo=PyTorch&logoColor=white)
- ![Flask](https://img.shields.io/badge/-Flask-000000?style=flat-square&logo=Flask&logoColor=white)
- ![Transformers](https://img.shields.io/badge/-Transformers-FFD000?style=flat-square&logo=Transformers&logoColor=black)
- ![Git LFS](https://img.shields.io/badge/-Git%20LFS-F64935?style=flat-square&logo=Git&logoColor=white)

## Setup and Installation

1. **Clone the repository**
   ```
   git clone https://github.com/beavishead/similar_images_search.git
   cd similar_images_search
   ```

2. **Set up a virtual environment**
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```

3. **Install dependencies**
   ```
   pip install -r requirements.txt

4. **Run the application**
   ```
   python app.py
   ```

5. **Access the application**
   - Open a web browser and go to `http://localhost:5000`

## Project Structure

- `app.py`: Main Flask application
- `model.py`: Contains the model loading and feature extraction logic
- `feature_extractor.py`: Script for pre-extracting features from the dataset
- `model_weights.pth`: Pre-trained model weights
- `pre_extracted_features.npz`: Pre-computed features for the dataset

## Future Improvements

- Deploy the app on web. The issues I faced are the dataset of images are being compared to is too large for the free tiers og deployment platforms.

- Make more meaningful app. The current app was made primarily as a blueprint of how a ml-model maybe implemented in a web application.

- More meaningful tasks would be:
-- scraping images from the web
-- detecting the potentially explicit content
-- adding the ability to label images with a certain textual category

- make the uploaded by users images disappear after some time