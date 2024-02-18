from flask import Flask, render_template, request, redirect, url_for
import cv2
import os
import base64
from werkzeug.utils import secure_filename

app = Flask(__name__)

UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'jpg', 'jpeg', 'png', 'gif'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def detect_faces(image_path):
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x,y,w,h) in faces:
        cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)

    screen_width = 1500  # Set your desired screen width
    aspect_ratio = img.shape[1] / img.shape[0]
    screen_height = int(screen_width / aspect_ratio)

    img = cv2.resize(img, (screen_width, screen_height))

    cv2.imshow('img',img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    _, encoded_img = cv2.imencode('.jpg', img)
    result_base64 = base64.b64encode(encoded_img.tobytes()).decode('utf-8')

    return len(faces), result_base64

@app.route('/')
def index():
    return render_template('index_faces.html')

@app.route('/', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return redirect(request.url)

    file = request.files['file']

    if file.filename == '':
        return redirect(request.url)

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)

        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        num_faces, result_path = detect_faces(filepath)

        return {
            "number_of_faces": num_faces,
            "image_path": filename,
            "result_path": os.path.basename(result_path)
        }
        # return render_template('index_faces.html', image_path=filename, result_path=os.path.basename(result_path), num_faces=num_faces)

if __name__ == '__main__':
    app.run(port="5050", debug=True)
