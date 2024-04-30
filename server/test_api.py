import requests

# API endpoint - change this based on the ip address shown when you run the app.py; Do not give localhost api, give the one using your ip address.
url = 'http://10.0.9.175:5000/predict'

# Path to the image file
image_path = r"D:\fw\Training\tdma\IMG_20240122_162414008.jpg"

# Read the image file
with open(image_path, 'rb') as file:
    # Prepare the POST request with the image file
    files = {'image': file}
    # Send the POST request
    response = requests.post(url, files=files)

# Check the response
if response.status_code == 200:
    # If server returns an OK response, print the predicted labels
    data = response.json()
    predicted_labels = data['predicted_labels']
    print('Predicted Labels:', predicted_labels)
else:
    # If the server did not return a 200 OK response, print the error message
    print('Error:', response.text)
