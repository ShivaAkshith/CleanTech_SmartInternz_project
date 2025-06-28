from flask import Flask, render_template, request, jsonify, url_for, redirect
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from PIL import Image
import numpy as np
import os
import tensorflow as tf

app = Flask(__name__)
model = tf.keras.models.load_model('waste_classification_model (1).h5')

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/about")
def about():
    return render_template("about.html")

@app.route("/contact")
def contact():
    return render_template("contact.html")

@app.route("/predict")
def predict():
    return render_template("predict.html")


@app.route("/portfolio", methods=['POST'])
def output():
    f = request.files['pc_image']
    img_path = os.path.join("static/uploads", f.filename)
    f.save(img_path)

    img = load_img(img_path, target_size=(224, 224))
    image_array = np.array(img)
    image_array = np.expand_dims(image_array, axis=0)

    pred = np.argmax(model.predict(image_array), axis=1)
    index = ['Biodegradable Images (0)', 'Recyclable Images (1)', 'Trash Images (2)']
    prediction = index[int(pred[0])]

    return render_template("portfolio-details.html", prediction=prediction, filename=f.filename)


if __name__=="__main__":
    app.run(debug=True)
