from flask import Flask, render_template, request, jsonify, send_from_directory, url_for
from werkzeug.utils import secure_filename
from model import download_file, extract_features_vit, get_similar_images
import os

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['DATASET_FOLDER'] = 'static/dataset'
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg', 'gif'}

# Google Drive file IDs - replace with your actual file IDs
MODEL_WEIGHTS_ID = "https://drive.google.com/file/d/1zDRHwzKKu-6hxyUdBgcuROGmwG1ydFBr/view?usp=sharing"
DATASET_ZIP_ID = "https://drive.google.com/file/d/1K3KFek9t9nqqQFzewyT85VqzfJ92TF6j/view?usp=sharing"

MODEL_WEIGHTS_PATH = "model_weights.pth"
DATASET_ZIP_PATH = "pre_extracted_features.npz"

@app.before_first_request
def initialize():
    download_file(MODEL_WEIGHTS_ID, MODEL_WEIGHTS_PATH)
    download_file(DATASET_ZIP_ID, DATASET_ZIP_PATH)
    # Any other initialization code you need

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

@app.route('/dataset/<path:filename>')
def serve_dataset_image(filename):
    return send_from_directory(app.config['DATASET_FOLDER'], filename)

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
            
            similar_images = get_similar_images(filepath)
            
            similar_image_data = [
                {
                    'path': url_for('serve_dataset_image', filename=os.path.basename(img['path'])),
                    'similarity': img['similarity'] * 100  # Convert similarity to percentage
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