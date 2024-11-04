from flask import Flask, request, jsonify, render_template, send_from_directory
import cv2
import numpy as np
import joblib
import os

app = Flask(__name__)

# Paths to your model and PCA files
MODEL_PATH = 'Image_Classifier.pkl'
PCA_PATH = 'pca_normal.pkl'

# Load your trained model and PCA model
best_estimator_svm = joblib.load(MODEL_PATH)
pca_normal = joblib.load(PCA_PATH)

# Function to preprocess a single image
def preprocess_single_image_standard(image_path, size=(128, 128)):
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Image file does not exist at the specified path: {image_path}")

    resized_image = cv2.resize(image, size)
    flattened_image = resized_image.flatten()
    pca_transformed_image = pca_normal.transform([flattened_image])

    return pca_transformed_image

# Function to predict the label of a single image
def predict_image_label_standard(image_path, model):
    preprocessed_image = preprocess_single_image_standard(image_path)
    predicted_label = model.predict(preprocessed_image)

    return predicted_label[0]

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/submit', methods=['POST'])
def submit():
    if 'my_image' not in request.files:
        return jsonify({'error': 'No file part'})

    file = request.files['my_image']

    if file.filename == '':
        return jsonify({'error': 'No selected file'})

    if file:
        # Save the uploaded file to the uploads directory
        image_path = os.path.join('uploads', file.filename)
        file.save(image_path)

        # Debugging: print the image path
        print(f"Image saved at: {image_path}")

        try:
            predicted_label = predict_image_label_standard(image_path, best_estimator_svm)

            # Debugging: print the prediction
            print(f"Predicted label: {predicted_label}")

            # Convert predicted_label to string (if necessary)
            predicted_label_str = predicted_label.decode('utf-8') if isinstance(predicted_label, bytes) else predicted_label

            return jsonify({'prediction': predicted_label_str, 'img_path': file.filename})

        except FileNotFoundError as e:
            # Debugging: print the error
            print(f"Error: {e}")
            return jsonify({'error': str(e)})

# Route to serve uploaded images
@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory('uploads', filename)

if __name__ == '__main__':
    app.run(debug=True)
