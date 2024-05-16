from flask import Flask
import base64
import cv2
import numpy as np
from flask_cors import CORS
from tensorflow.keras.models import load_model

app = Flask(__name__)

cors = CORS(app)

IMAGE_SIZE = 128

Art_Categories = ['abstract','genre-painting','landscape', 'portrait']


model = load_model('models/checkpoint5.model.keras')

@app.route("/")
def hello_world():
  return "<p>Hello, World!</p>"

from flask import request, jsonify

@app.route('/upload-image', methods=['POST'])
def upload_image():
  image_data = request.json.get('image')
  if not image_data:
    return jsonify({'error': 'Missing image data'}), 400

  img_arr = readb64(image_data)
  prediction_result = int(prediction(img_arr))
  # print(prediction_result, type(prediction_result))
  return jsonify({'prediction': prediction_result}), 200
                #  np.argmax(prediction_result)]

def readb64(uri):
   encoded_data = uri.split(',')[1]
   nparr = np.fromstring(base64.b64decode(encoded_data), np.uint8)
   img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
   return img

def prediction(img_arr):
  # img_array=cv2.imread(os.path.join(path,img))
  img_arr=cv2.resize(img_arr,(IMAGE_SIZE,IMAGE_SIZE))
  img_arr = img_arr / 255.0

    # Predict the values from the test dataset
  Y_pred = model.predict(img_arr.reshape(1,IMAGE_SIZE,IMAGE_SIZE,3))
  Y_pred = np.argmax(Y_pred, axis=1)
  print(Y_pred)
  return Y_pred[0]
  