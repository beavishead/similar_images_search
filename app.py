from flask import Flask, render_template, request, jsonify, send_from_directory, url_for
from werkzeug.utils import secure_filename
from model import get_similar_images
import os

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['DATASET_FOLDER'] = 'static/dataset'  # Adjust this path as needed
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg', 'gif'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

@app.route('/dataset/<path:filename>')
def serve_dataset_image(filename):
    print(f"Attempting to serve image: {filename}")  # Debug print
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
            
            # Convert relative paths to URLs and include similarity percentages
            similar_image_data = [
                {
                    'path': url_for('serve_dataset_image', filename=os.path.basename(img['path'])),
                    'similarity': img['similarity'] * 100  # Convert similarity to percentage
                }
                for img in similar_images
            ]
            
            print("Similar image data:", similar_image_data)  # Debug print
            
            return jsonify({'similar_images': similar_image_data})
        else:
            return jsonify({'error': 'File type not allowed'})
    
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)