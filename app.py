import os
import logging
from flask import Flask, render_template, request, jsonify, send_from_directory, url_for
from werkzeug.utils import secure_filename
from model import load_model, load_features, get_similar_images

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['DATASET_FOLDER'] = 'static/dataset'
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg', 'gif'}

# File paths (adjust these if using Render's Asset Storage)
MODEL_WEIGHTS_PATH = "model_weights.pth"
FEATURES_PATH = "pre_extracted_features.npz"
DATASET_FOLDER = app.config['DATASET_FOLDER']

logging.basicConfig(level=logging.INFO)

dataset_features = None
image_paths = None
is_initialized = False

def initialize():
    global dataset_features, image_paths, is_initialized
    if not is_initialized:
        try:
            logging.info("Starting initialization...")
            
            # Load model
            if os.path.exists(MODEL_WEIGHTS_PATH):
                load_model(MODEL_WEIGHTS_PATH)
            else:
                raise FileNotFoundError(f"Model weights file not found: {MODEL_WEIGHTS_PATH}")
            
            # Load features
            if os.path.exists(FEATURES_PATH):
                dataset_features, image_paths = load_features(FEATURES_PATH)
            else:
                raise FileNotFoundError(f"Pre-extracted features file not found: {FEATURES_PATH}")
            
            logging.info("Initialization complete")
            is_initialized = True
        except Exception as e:
            logging.error(f"Initialization error: {str(e)}")
            raise

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
    return send_from_directory(DATASET_FOLDER, filename)

@app.route('/', methods=['GET', 'POST'])
def index():
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