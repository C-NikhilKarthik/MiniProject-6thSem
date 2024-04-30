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
from ultralytics import YOLO


app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})  # Enable CORS for all origins

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
    print(image_copy)
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
















def get_yolo_faces(image):
    model = YOLO('yolov8n-face.pt')
    print("Yolov8 model loaded successfully")
    print("image loaded")

    if image is not None:
        print("here")
        # image = cv2.resize(image, (640, 640))

        print("image resized")
        faces=[]
        results = model(image)
        print("here1")
        for result in results:
            print("show ")

            for box in result.boxes.xywh:
                        box = box.cpu().numpy().tolist()
                        box = np.around(box, decimals=4).tolist()
                        x,y,w,h=int(box[0]),int(box[1]),int(box[2]),int(box[3])
                        # face_image = image[y:y + h, x:x + w]
                        # face_image = cv2.resize(face_image, (200, 200))
                        # # predictions=model2(face_image)
                        # # print(predictions)
                        # cv2.imshow("Processed Face", face_image)
                        # cv2.waitKey(0)
                        faces.append([x,y,w,h])

        return faces
    else:
        print(f"Error: Unable to load image '{image}'.")
        return None









def detect_faces(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_detector.detectMultiScale(gray, 1.1, 4)
    print(">>>",faces[0])
    return faces

def get_preds(face_images):
    embeddings = []
    predictions = []
    for i, face_image in enumerate(face_images):
        cv2.imshow("get_preds", face_image)
        cv2.waitKey(0)
        preds = get_predictions(image_copy=face_image,label_encoder=label_encoder,model_thr=model_thr,models=models, tflite_model=tflite_model)
        print("Predictions: ", preds)
        
        predictions.append("Face-"+str(i)+": "+', '.join(preds))
    print("Predictions: ", predictions)
    return predictions

def predict(image):
    # Predict labels for the image
    # faces = detect_faces(image)
    faces = get_yolo_faces(image=image)
    face_images = [image[y-h//2:y+h//2, x-w//2:x+w//2] for (x, y, w, h) in faces]
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

        return jsonify({'predicted_labels': prediction}), 200
    
    except Exception as e:
        print(e)
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True, use_reloader = False)
