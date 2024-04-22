from flask import Flask, request, jsonify
import cv2
import numpy as np
from keras_facenet import FaceNet
import joblib
from flask_cors import CORS
import csv
import os
from datetime import datetime

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})  # Enable CORS for all origins

# Load models
face_detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
embedder = FaceNet()
loaded_model = joblib.load("cnn_model.pkl")
loaded_label_encoder = joblib.load("label_encoder.pkl")

def detect_faces(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_detector.detectMultiScale(gray, 1.1, 4)
    return faces

def extract_embeddings(face_images):
    embeddings = []
    for face_image in face_images:
        resized_image = cv2.resize(face_image, (160, 160))
        resized_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2RGB)
        input_image = np.expand_dims(resized_image, axis=0)
        embedding = embedder.embeddings(input_image)
        embeddings.append(embedding)
    return embeddings

def predict(image):
    # Predict labels for the image
    faces = detect_faces(image)
    face_images = [image[y:y+h, x:x+w] for (x, y, w, h) in faces]
    embeddings = extract_embeddings(face_images)
    predicted_labels = [loaded_model.predict(embedding) for embedding in embeddings]
    return predicted_labels

def add_roll_numbers_to_csv(date, roll_numbers):
    # Check if the CSV file exists
    file_exists = os.path.exists('data.csv')

    with open('data.csv', mode='a', newline='') as file:
        fieldnames = ['Date', 'Roll Numbers']
        writer = csv.DictWriter(file, fieldnames=fieldnames)

        # If the file doesn't exist, write header
        if not file_exists:
            writer.writeheader()

        # Check if the date already exists in the CSV
        date_exists = False
        with open('data.csv', mode='r') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                if row['Date'] == date:
                    date_exists = True
                    break

        # If the date exists, append roll numbers to the existing row
        if date_exists:
            with open('data.csv', mode='r') as csvfile:
                reader = csv.DictReader(csvfile)
                rows = list(reader)
            with open('data.csv', mode='w', newline='') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                for row in rows:
                    if row['Date'] == date:
                        # Split existing roll numbers and convert to set for uniqueness
                        existing_roll_numbers = set(row['Roll Numbers'].split(', '))
                        # Add new roll numbers to the set
                        existing_roll_numbers.update(roll_numbers)
                        row['Roll Numbers'] = ', '.join(existing_roll_numbers)
                    writer.writerow(row)
        # If the date does not exist, create a new row with the date and roll numbers
        else:
            writer.writerow({'Date': date, 'Roll Numbers': ', '.join(roll_numbers)})


@app.route('/api', methods=['GET'])
def return_ascii():
    d = {}
    input_chr = str(request.args['query'])
    answer = str(ord(input_chr))
    d['output'] = answer
    d['result'] = "21bcs052"
    return jsonify(d)

@app.route('/', methods=["GET"])
def hello():
    d = {}
    d['pred'] = "test"
    return jsonify(d)

@app.route('/predict', methods=['POST'])
def predict_endpoint():
    try:
        # Check if image file is present in the request
        if 'image' not in request.files:
            print("error!!")
            return jsonify({'error': 'No image provided'}), 400
        
        # Read image file from request
        image_file = request.files['image']
        image_np = np.frombuffer(image_file.read(), np.uint8)
        image = cv2.imdecode(image_np, cv2.IMREAD_COLOR)

        # Predict labels for the image
        prediction = predict(image)

        # Format and return prediction
        labels = []
        for i in prediction:
            print(i)
            predicted_label_index = np.argmax(i)
            if predicted_label_index <= 10 and i[0][predicted_label_index]>=0.95:
                predicted_label = loaded_label_encoder.inverse_transform([predicted_label_index])[0]
            else:
                predicted_label = "UNKNOWN"
            labels.append(predicted_label)
        unique_labels = list(set(labels))
        print(unique_labels)
        return jsonify({'predicted_labels': unique_labels}), 200
    
    except Exception as e:
        print(e)
        return jsonify({'error': str(e)}), 500
    
@app.route('/store', methods=['POST'])
def store_data():
    try:
        if 'data' not in request.json:
            print('error!!')
            return jsonify({'error': 'No data provided in the request'}), 400
        
        date = datetime.now().strftime('%Y-%m-%d')
        
        roll_numbers = request.json['data']
        
        add_roll_numbers_to_csv(date, roll_numbers)

        return jsonify({'message': 'Data stored successfully'})
    
    except Exception as e:
        print(e)
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True, use_reloader = False)
