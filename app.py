from flask import Flask, render_template, request, redirect, url_for
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import os
import cv2  

app = Flask(__name__)
MODEL_PATH = 'plant_disease_model.h5'  
model = load_model(MODEL_PATH)


CLASS_LABELS = [
    'Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy',
    'Blueberry___healthy', 'Cherry_(including_sour)___Powdery_mildew', 'Cherry_(including_sour)___healthy',
    'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot', 'Corn_(maize)___Common_rust_',
    'Corn_(maize)___Northern_Leaf_Blight', 'Corn_(maize)___healthy', 'Grape___Black_rot',
    'Grape___Esca_(Black_Measles)', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 'Grape___healthy',
    'Orange___Haunglongbing_(Citrus_greening)', 'Peach___Bacterial_spot', 'Peach___healthy',
    'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy', 'Potato___Early_blight',
    'Potato___Late_blight', 'Potato___healthy', 'Raspberry___healthy', 'Soybean___healthy',
    'Squash___Powdery_mildew', 'Strawberry___Leaf_scorch', 'Strawberry___healthy',
    'Tomato___Bacterial_spot', 'Tomato___Early_blight', 'Tomato___Late_blight',
    'Tomato___Leaf_Mold', 'Tomato___Septoria_leaf_spot', 'Tomato___Spider_mites Two-spotted_spider_mite',
    'Tomato___Target_Spot', 'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus',
    'Tomato___healthy'
]


UPLOAD_FOLDER = 'static/uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


history_data = []


def is_leaf_image(image_path):
    img = cv2.imread(image_path)
    if img is None:
        return False  
    hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    green_mask = cv2.inRange(hsv_img, (36, 25, 25), (86, 255, 255))
    green_ratio = np.sum(green_mask) / (img.shape[0] * img.shape[1])
    return green_ratio > 0.2 

@app.route('/')
def home():
    return render_template('index.html', title="Plant Disease Detection")

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return render_template(
            'index.html',
            result="No file part.",
            title="Invalid Image",
            show_history_button=False
        )
    file = request.files['file']
    if file.filename == '':
        return render_template(
            'index.html',
            result="No selected file.",
            title="Invalid Image",
            show_history_button=False
        )
    if file:
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(file_path)

        if not is_leaf_image(file_path):
            os.remove(file_path)  
            return render_template(
                'index.html',
                result="Please upload a valid  image.",
                title="Invalid Image",
                show_history_button=False
            )

        img = image.load_img(file_path, target_size=(224, 224))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0) / 255.0

        predictions = model.predict(img_array)
        predicted_class = np.argmax(predictions, axis=1)[0]
        result = CLASS_LABELS[predicted_class]

        history_data.append({'image': file.filename, 'result': result})

        return render_template(
            'index.html',
            result=result,
            image_url=file_path,
            title="Prediction Result",
            show_history_button=True  
        )

@app.route('/history')
def history():
    return render_template('history.html', history=history_data, title="Prediction History")

if __name__ == '__main__':
    app.run(debug=True) 