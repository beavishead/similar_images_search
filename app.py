from flask import Flask, render_template, request, jsonify, send_from_directory, url_for
from werkzeug.utils import secure_filename
from model import download_file, load_model, load_features, get_similar_images
import os
import logging
import zipfile

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['DATASET_FOLDER'] = 'static/dataset'
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg', 'gif'}

MODEL_WEIGHTS_ID = "1nC1I9f_bj66F18fM2b0e_aC8snwHVjas"
FEATURES_ZIP_ID = "1zDRHwzKKu-6hxyUdBgcuROGmwG1ydFBr"
DATASET_ZIP_ID = "1K3KFek9t9nqqQFzewyT85VqzfJ92TF6j"  # Add this ID for your dataset zip

MODEL_WEIGHTS_PATH = "model_weights.pth"
FEATURES_PATH = "pre_extracted_features.npz"
DATASET_ZIP_PATH = "dataset.zip"
DATASET_EXTRACT_PATH = app.config['DATASET_FOLDER']

logging.basicConfig(level=logging.INFO)

dataset_features = None
image_paths = None
is_initialized = False

def safe_extract_zip(zip_path, extract_path):
    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_path)
        logging.info(f"Extracted {zip_path} to {extract_path}")
    except zipfile.BadZipFile:
        logging.error(f"File is not a valid zip file: {zip_path}")
    except Exception as e:
        logging.error(f"Error extracting {zip_path}: {str(e)}")

def initialize():
    global dataset_features, image_paths, is_initialized
    if not is_initialized:
        try:
            logging.info("Starting initialization...")
            
            # Download files if they don't exist
            if not os.path.exists(MODEL_WEIGHTS_PATH):
                download_file(MODEL_WEIGHTS_ID, MODEL_WEIGHTS_PATH)
            if not os.path.exists(FEATURES_PATH):
                download_file(FEATURES_ZIP_ID, FEATURES_PATH)
            if not os.path.exists(DATASET_ZIP_PATH):
                download_file(DATASET_ZIP_ID, DATASET_ZIP_PATH)
            
            # Extract dataset if needed
            if os.path.exists(DATASET_ZIP_PATH) and not os.listdir(DATASET_EXTRACT_PATH):
                safe_extract_zip(DATASET_ZIP_PATH, DATASET_EXTRACT_PATH)
            
            # Load model and features
            if os.path.exists(MODEL_WEIGHTS_PATH):
                load_model(MODEL_WEIGHTS_PATH)
            else:
                raise FileNotFoundError(f"Model weights file not found: {MODEL_WEIGHTS_PATH}")
            
            if os.path.exists(FEATURES_PATH):
                dataset_features, image_paths = load_features(FEATURES_PATH)
            else:
                raise FileNotFoundError(f"Pre-extracted features file not found: {FEATURES_PATH}")
            
            logging.info("Initialization complete")
            is_initialized = True
        except Exception as e:
            logging.error(f"Initialization error: {str(e)}")
            raise  # Re-raise the exception to prevent the app from starting if initialization fails

@app.before_request
def before_request():
    if not is_initialized:
        try:
            initialize()
        except Exception as e:
            logging.error(f"Initialization failed: {str(e)}")
            return "Application failed to initialize. Please check the logs.", 500

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

@app.route('/dataset/<path:filename>')
def serve_dataset_image(filename):
    return send_from_directory(app.config['DATASET_FOLDER'], filename)

@app.route('/', methods=['GET', 'POST'])
def index():
    if not is_initialized:
        return "Application is initializing. Please wait and refresh in a few moments.", 503
    
    if request.method == 'POST':
        if 'file' not in request.files:
            return jsonify({'error': 'No file part'})
        
        file = request.files['file']
        
        if file.filename == '':
            return jsonify({'error': 'No selected file'})
        
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            similar_images = get_similar_images(filepath, dataset_features, image_paths)
            
            similar_image_data = [
                {
                    'path': url_for('serve_dataset_image', filename=os.path.basename(img['path'])),
                    'similarity': img['similarity'] * 100
                }
                for img in similar_images
            ]
            
            return jsonify({'similar_images': similar_image_data})
        else:
            return jsonify({'error': 'File type not allowed'})
    
    return render_template('index.html')

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)