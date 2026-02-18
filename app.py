from flask import Flask, render_template, request
from tensorflow.keras.preprocessing import image
import tensorflow as tf
import numpy as np
import os

app = Flask(__name__)

# ✅ Correct SavedModel loading
model = tf.saved_model.load('pneumonia_savedmodel')
infer = model.signatures['serving_default']

IMG_SIZE = 224

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():

    if 'file' not in request.files:
        return render_template('index.html', prediction="No file uploaded")

    file = request.files['file']

    if file.filename == '':
        return render_template('index.html', prediction="No file selected")

    os.makedirs('static', exist_ok=True)

    filepath = os.path.join('static', file.filename)
    file.save(filepath)

    img = image.load_img(filepath, target_size=(IMG_SIZE, IMG_SIZE))
    img_array = image.img_to_array(img)
    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # ✅ Proper inference call
    prediction = infer(tf.constant(img_array))['output_0'][0][0]

    prediction = float(prediction)

    if prediction > 0.7:
        result = "PNEUMONIA DETECTED"
        confidence = round(prediction * 100, 2)
    else:
        result = "NORMAL"
        confidence = round((1 - prediction) * 100, 2)

    return render_template(
        'index.html',
        prediction=result,
        confidence=confidence,
        img_path=filepath
    )

if __name__ == '__main__':
    app.run(debug=True)
