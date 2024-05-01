import json
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import pairwise_distances
import tensorflow as tf
from flask import Flask, request, jsonify
import cv2
import numpy as np
import joblib
from flask_cors import CORS
import pandas as pd
import jwt
import psycopg2
from flask_bcrypt import Bcrypt

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})  # Enable CORS for all origins
bcrypt = Bcrypt(app)

# Global constants
global Email
global conn
app.config["SECRET_KEY"] = "jshefhbkj5456789654vgbdjgf"

conn = psycopg2.connect(
    host="ep-silent-cake-a1g9wkzh.ap-southeast-1.aws.neon.tech",
    database="snapmark",
    user="snapmark_owner",
    password='wGKHm9vzX1In'
)

def connectDB():
    global conn
    cur = conn.cursor()
    cur.execute('''CREATE TABLE IF NOT EXISTS USERS(
                id uuid DEFAULT uuid_generate_v4() PRIMARY KEY,
                email varchar(255) UNIQUE,
                name varchar(255),
                password varchar(255)
    );''')
    conn.commit()
    cur.close()
# Load models
face_detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')


with open("reversed_obj_ids.json", "r") as file:
    obj_id_to_usn = json.load(file)
# print(obj_id_to_usn)

class CustomKNN(KNeighborsClassifier):
    def __init__(self, n_neighbors=5, **kwargs):
        super().__init__(n_neighbors=n_neighbors, **kwargs)

    def predict(self, X):
        labels = super().predict(X)
        distances = pairwise_distances(X, self._fit_X, metric='euclidean', n_jobs=-1).min(axis=1)
        nearest_neighbor_index = np.argmin(pairwise_distances(X, self._fit_X, metric='euclidean', n_jobs=-1), axis=1)
        return labels, distances, nearest_neighbor_index

with open("reversed_obj_ids.json", "r") as file:
    obj_id_to_usn = json.load(file)

with open("obj.names", "r") as file:
    values = [int(line.strip()) for line in file]

custom_dict = {i: obj_id_to_usn[str(values[i])] for i in range(len(values))}

def load_tflite_model(file = "facenet.tflite"):
    interpreter = tf.lite.Interpreter(model_path=file)
    interpreter.allocate_tensors()
    return interpreter

def preprocess_image(image, image_size=(160, 160), kernel_size=(5, 5), sigma_x=0.5, sigma_y=0.5):
    resized_image = cv2.resize(image, image_size)
    resized_image = cv2.GaussianBlur(resized_image, kernel_size, sigma_x, sigma_y)
    resized_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2RGB)
    resized_image = resized_image.astype('float32')
    mean, std = resized_image.mean(), resized_image.std()
    resized_image = (resized_image - mean) / std
    return resized_image

def tflite_predict(face_model, image):
    input_details = face_model.get_input_details()
    output_details = face_model.get_output_details()
    input_shape = input_details[0]['shape']
    input_data = image.reshape(input_shape)
    face_model.set_tensor(input_details[0]['index'], input_data)
    face_model.invoke()
    output = []
    output_data = face_model.get_tensor(output_details[0]['index'])
    for i in output_data:
        output.append(i)
    return np.stack(output)

def get_tflite_facenet_embedding(image,tflite_model, image_size = (160,160), kernel_size=(3,3), sigma_x=0.5, sigma_y=0.5):
    preprocessed_image = preprocess_image(image=image)
    embedding = tflite_predict(face_model=tflite_model, image=preprocessed_image)
    curr_embedding = np.array(embedding)
    return curr_embedding

def get_models_prediction(models, model_thr, embedding,label_encoder,debug=0):
    knn_pred = ""
    rf_pred = ""
    lr_pred = ""
    svm_l_pred = ""
    svm_rbf_pred = ""
    svm_ploy_pred = ""
    svm_sig_pred = ""
    ann1_pred = ""
    ann2_pred = ""
    cnn1_pred = ""
    cnn2_pred = ""
    
    for i,pair in enumerate(model_thr):
        prediction = ""
        model = models[i]
        if pair[0]=="models-pkl/custom_knn_model.pkl":
            result = model.predict(embedding)
            pred = result[0][0]
            distance = result[1][0]
            prediction=""
            if distance>pair[1]: 
                prediction = obj_id_to_usn["100"]
            else: 
                prediction = obj_id_to_usn[str(pred)]

        if pair[0]=="models-pkl/random_forest_model.pkl":
            result = model.predict(embedding) [0]
            decision_score = model.predict_proba(embedding)
            prediction=obj_id_to_usn[str(result)]
            if np.max(decision_score[0])<pair[1]: 
                prediction="UNKNOWN"
            
        if pair[0]=="models-pkl/logistic_regression_model.pkl" or pair[0]=="models-pkl/svm_linear_model.pkl" or pair[0]=="models-pkl/svm_rbf_model.pkl" or pair[0]=="models-pkl/svm_poly_model.pkl" or pair[0]=="models-pkl/svm_sigmoid_model.pkl":
            result = model.predict(embedding) [0]
            decision_score = model.decision_function(embedding)
            prediction=obj_id_to_usn[str(result)]
            if np.max(decision_score[0])<pair[1]: 
                prediction="UNKNOWN"

        if pair[0] == "models-pkl/ann1_model.pkl" or pair[0] == "models-pkl/ann2_model.pkl" or pair[0] == "models-pkl/cnn1_model.pkl" or pair[0]=="models-pkl/cnn2_model.pkl":
            result = model.predict(embedding) [0]
            result_max_index = np.argmax(result)
            if(np.max(result)<pair[1]): 
                prediction="UNKNOWN"
            else:
                prediction=obj_id_to_usn[str(label_encoder.inverse_transform([result_max_index])[0])]

        if pair[0]=="models-pkl/custom_knn_model.pkl":    
            knn_pred = prediction
        if pair[0]=="models-pkl/random_forest_model.pkl":    
            rf_pred = prediction
        if pair[0]=="models-pkl/logistic_regression_model.pkl":    
            lr_pred = prediction
        if pair[0]=="models-pkl/svm_linear_model.pkl":    
            svm_l_pred = prediction
        if pair[0]=="models-pkl/svm_rbf_model.pkl":    
            svm_rbf_pred = prediction
        if pair[0]=="models-pkl/svm_poly_model.pkl":    
            svm_ploy_pred = prediction
        if pair[0]=="models-pkl/svm_sigmoid_model.pkl":    
            svm_sig_pred = prediction
        if pair[0]=="models-pkl/ann1_model.pkl":    
            ann1_pred = prediction
        if pair[0]=="models-pkl/ann2_model.pkl":    
            ann2_pred = prediction
        if pair[0]=="models-pkl/cnn1_model.pkl":    
            cnn1_pred = prediction
        if pair[0]=="models-pkl/cnn2_model.pkl":
            cnn2_pred = prediction

    return [knn_pred,rf_pred,lr_pred,svm_l_pred,svm_rbf_pred,svm_ploy_pred,svm_sig_pred,ann1_pred,ann2_pred,cnn1_pred,cnn2_pred]

def get_predictions(image_copy, models, model_thr, label_encoder, tflite_model):
    image = cv2.resize(image_copy,(160,160))

    embedding = get_tflite_facenet_embedding(image=image, tflite_model=tflite_model)

    knn_pred, rf_pred, lr_pred, svm_l_pred, svm_rbf_pred, svm_ploy_pred, svm_sig_pred, ann1_pred, ann2_pred, cnn1_pred, cnn2_pred = get_models_prediction(models=models, model_thr=model_thr, embedding=embedding, label_encoder=label_encoder)
    
    return [knn_pred, rf_pred, lr_pred, svm_l_pred, svm_rbf_pred, svm_ploy_pred, svm_sig_pred, ann1_pred, ann2_pred, cnn1_pred, cnn2_pred]

model_thr=  [
    ("models-pkl/custom_knn_model.pkl", 0.78),
    ("models-pkl/random_forest_model.pkl", 0.16),
    ("models-pkl/logistic_regression_model.pkl", 2.8),
    ("models-pkl/svm_linear_model.pkl", 31.35),
    ("models-pkl/svm_rbf_model.pkl", 32.3),
    ("models-pkl/svm_poly_model.pkl", 30),
    ("models-pkl/svm_sigmoid_model.pkl", 31.35),
    ("models-pkl/ann1_model.pkl", 0.35),
    ("models-pkl/ann2_model.pkl", 0.30),
    ("models-pkl/cnn1_model.pkl", 0.35),
    ("models-pkl/cnn2_model.pkl", 0.55)
]

model_knn = joblib.load("models-pkl/custom_knn_model.pkl")
model_rf = joblib.load("models-pkl/random_forest_model.pkl")
model_lr = joblib.load("models-pkl/logistic_regression_model.pkl")
model_svm_l = joblib.load("models-pkl/svm_linear_model.pkl")
model_svm_rbf = joblib.load("models-pkl/svm_rbf_model.pkl")
model_svm_poly = joblib.load("models-pkl/svm_poly_model.pkl")
model_svm_sig = joblib.load("models-pkl/svm_sigmoid_model.pkl")
model_ann1 = joblib.load("models-pkl/ann1_model.pkl")
model_ann2 = joblib.load("models-pkl/ann2_model.pkl")
model_cnn1 = joblib.load("models-pkl/cnn1_model.pkl")
model_cnn2 = joblib.load("models-pkl/cnn2_model.pkl")
models = [model_knn,
              model_rf,
              model_lr,
              model_svm_l,
              model_svm_rbf,
              model_svm_poly,
              model_svm_sig,
              model_ann1,
              model_ann2,
              model_cnn1,
              model_cnn2]
tflite_model = load_tflite_model()

label_encoder = "models-pkl/le.pkl"
label_encoder = joblib.load(label_encoder)

def detect_faces(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_detector.detectMultiScale(gray, 1.1, 4)
    print(">>>",faces[0])
    return faces

def get_preds(face_images):
    embeddings = []
    predictions = []
    for i, face_image in enumerate(face_images):
        # cv2.imshow("kar", face_image)
        # cv2.waitKey(0)
        preds = get_predictions(image_copy=face_image,label_encoder=label_encoder,model_thr=model_thr,models=models, tflite_model=tflite_model)
        print("Predictions: ", preds)
        
        predictions.append(preds)
    print("Predictions: ", predictions)
    return predictions

def predict(image):
    # Predict labels for the image
    faces = detect_faces(image)
    face_images = [image[y:y+h, x:x+w] for (x, y, w, h) in faces]
    predicted_labels = get_preds(face_images)
    return predicted_labels

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

predictions = []

@app.route('/register', methods=["POST"])
def registration():
    data = request.json
    email = data.get('email')
    name = data.get('name')
    password = data.get('password')

    # Check if email already exists
    cur = conn.cursor()
    cur.execute('SELECT * FROM USERS WHERE email = %s', (email,))
    existing_user = cur.fetchone()
    cur.close()

    if existing_user:
        return jsonify({'message': 'Email already exists', 'status': False})

    # Hash the password
    hashed_password = bcrypt.generate_password_hash(password).decode('utf-8')

    # Insert new user
    cur = conn.cursor()
    cur.execute( 
        '''INSERT INTO USERS
        (email, name, password) VALUES (%s, %s, %s)''', 
        (email, name, hashed_password)) 
    conn.commit()
    cur.close()

    response_data = {'message': 'Registration successful', 'status': True}
    return jsonify(response_data)

@app.route('/login', methods=["POST"])
def login():
    global Email
    data = request.json
    email = data.get('email')
    password = data.get('password')

    print(email,password)

    # Check if email already exists
    cur = conn.cursor()
    cur.execute('SELECT * FROM USERS WHERE email = %s', (email,))
    existing_user = cur.fetchone()
    cur.close()

    if not existing_user:
        return jsonify({'message': 'Account not exist.', "status": False})

    get_pass = existing_user[3]
    is_valid = bcrypt.check_password_hash(get_pass, password)

    if not is_valid:
        return jsonify({'message': 'Wrong Password', "status": False})

    # Generate JWT token
    token = jwt.encode(
        {"id": existing_user[0]},
        app.config["SECRET_KEY"],
        algorithm="HS256"
    )

    Email= email

    return jsonify({'message': 'Login Successfull', "status": True, 'token': token})

@app.route('/predict', methods=['POST'])
def predict_endpoint():
    global predictions
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
        for i in prediction:
            predictions.append(i)
        return jsonify({'predicted_labels': prediction}), 200
    
    except Exception as e:
        print(e)
        return jsonify({'error': str(e)}), 500
    
@app.route('/save_predictions', methods=['GET'])
def save_predictions():
    global predictions
    try:
        # Convert predictions to a JSON string
        json_string = json.dumps(predictions)
        
        # Write JSON string to a text file
        with open('predicted_labels.txt', 'w') as f:
            f.write(json_string)
        
        # Load existing Excel sheet if available
        try:
            df = pd.read_excel('predicted_labels.xlsx')
        except FileNotFoundError:
            df = pd.DataFrame(columns=['predicted_labels'])
        
        # Check if the pattern exists in the Excel sheet
        for item in predictions:
            # Filter out "UNKNOWN" values and calculate the mode of the remaining elements in the sublist
            filtered_item = [ele for ele in item if ele != "UNKNOWN"]
            if not filtered_item:
                continue  # Skip if all elements are "UNKNOWN"
            mode_value = max(set(filtered_item), key=filtered_item.count)
            
            # Check if item[2] matches the mode and is not already in the Excel sheet
            if item[2] == mode_value and item[2] not in df['predicted_labels'].values:
                # Add the pattern to the DataFrame
                new_row = pd.DataFrame({"predicted_labels": [item[2]]})
                df = pd.concat([df, new_row], ignore_index=True)
                print(f"Added pattern {item[2]} to the DataFrame.")
            else:
                print(f"Pattern {item[2]} either does not match the mode, contains only 'UNKNOWN' values, or already exists in the DataFrame.")
        
        # Save the updated Excel sheet
        df.to_excel('predicted_labels.xlsx', index=False)
        
        return jsonify({"message": "Predicted labels processed and Excel sheet updated"}), 200
    
    except Exception as e:
        print(e)
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True, use_reloader = True)
