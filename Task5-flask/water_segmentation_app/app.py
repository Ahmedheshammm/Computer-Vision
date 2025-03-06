from flask import Flask, request, render_template, jsonify
import os
import numpy as np
import tensorflow as tf
import utils 
from werkzeug.utils import secure_filename
from tensorflow.keras.layers import Layer, Conv2D, BatchNormalization, ReLU
from tensorflow.keras.models import Sequential

class ConvBlock(Layer):
    def __init__(self, filters=256, kernel_size=3, dilation_rate=1, **kwargs):
        super(ConvBlock, self).__init__(**kwargs)
        
        self.filters = filters
        self.kernel_size = kernel_size
        self.dilation_rate = dilation_rate
        
        self.net = Sequential([
            Conv2D(filters, kernel_size=kernel_size, padding='same', 
                  dilation_rate=dilation_rate, use_bias=False, 
                  kernel_initializer='he_normal'),
            BatchNormalization(), 
            ReLU()
        ])
    
    def call(self, X):
        return self.net(X)
    
    def get_config(self):
        base_config = super().get_config()
        return {
            **base_config,
            "filters": self.filters,
            "kernel_size": self.kernel_size,
            "dilation_rate": self.dilation_rate,
        }

app = Flask(__name__)

# Configure upload folder
UPLOAD_FOLDER = 'static/uploads'
ALLOWED_EXTENSIONS = {'tif', 'tiff'}  # Add other formats if needed
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Ensure upload directory exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load the model
model = tf.keras.models.load_model('models/water_segmentation_model.h5', 
                                 custom_objects={'ConvBlock': ConvBlock})

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/', methods=['GET'])
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'})
    
    if file and allowed_file(file.filename):
        # Save the uploaded file
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        try:
            # Preprocess the image
            image = utils.preprocess_image(filepath)
            
            # Make prediction
            prediction = model.predict(image)
            
            # Process the prediction
            result_path = utils.postprocess_prediction(prediction, filename)
            
            # Create overlay visualization
            overlay_path = utils.overlay_prediction(filepath, prediction, filename)
            
            return render_template('result.html', 
                                input_image=os.path.join('uploads', filename),
                                prediction_image=os.path.join('uploads', f'pred_{filename}.png'),
                                overlay_image=os.path.join('uploads', f'overlay_{filename}.png'))
        except Exception as e:
            return jsonify({'error': f'Error processing image: {str(e)}'})
    
    return jsonify({'error': 'Invalid file type'})

if __name__ == '__main__':
    app.run(debug=True)