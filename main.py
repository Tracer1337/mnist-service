import keras
from flask import Flask, request, make_response
import numpy as np
from PIL import Image
import cv2
import matplotlib.pyplot as plt
import os
import random
import string

model = keras.models.load_model("mnist_model.h5")

def predict(image):
  prediction = model.predict(image.reshape(-1, 28, 28))
  print(prediction)
  return np.argmax(prediction)

def format_image(img):
  img_array = cv2.imread(img, cv2.IMREAD_GRAYSCALE)
  img_pil = Image.fromarray(img_array)
  img_28x28 = np.array(img_pil.resize((28, 28), Image.ANTIALIAS))
  img_28x28 = np.invert(img_28x28)
  return img_28x28

def test_image(n):
  img_28x28 = format_image(f"number-{n}.png")
  print(img_28x28)
  plt.imsave(f'number-{n}_formatted.png', img_28x28)
  print(predict(img_28x28))

def random_filename(ext):
  return ''.join(random.choice(string.ascii_lowercase) for i in range(16)) + '.' + ext

app = Flask(__name__)

@app.route("/")
def health_check():
  return "Hello World"

@app.route("/predict", methods=["POST"])
def handle_predict():
  filename = random_filename('png')
  with open(filename, 'wb') as f:
    f.write(request.get_data())
  result = predict(format_image(filename))
  os.remove(filename)
  return str(result)

@app.route("/format", methods=["POST"])
def handle_format():
  filename = random_filename('png')
  with open(filename, 'wb') as f:
    f.write(request.get_data())
  formatted = format_image(filename)
  os.remove(filename)
  _, buffer = cv2.imencode('.png', formatted)
  response = make_response(buffer.tobytes())
  response.headers['Content-Type'] = 'image/png'
  return response

# test_image('7')
