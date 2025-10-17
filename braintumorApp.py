from flask import Flask, render_template, request, send_file
import numpy as np
import cv2
from keras.models import load_model
from keras.applications.efficientnet import preprocess_input
import os

app = Flask(__name__, static_folder='static', static_url_path='/static/')

prep = {0: 'No', 1: 'Yes'}

modal = load_model('model and data/Models.h5')

def predict_label(img_path):
    img = cv2.imread(img_path)
    img = cv2.resize(img, (224, 224))
    img = np.array(img)
    img_batch = np.expand_dims(img, axis=0)
    img_batch = preprocess_input(img_batch)  # Preprocess the image for EfficientNet
    pred_index = np.argmax(modal.predict(img_batch), axis=1)
    pred_class = prep[pred_index[0]]
    return "\n{}".format(pred_class)

@app.route("/", methods=['GET', 'POST'])
def main():
    return render_template("project.html")

@app.route("/about/", methods=['GET'])
def about_page():
    return """Hello there!!!"""

@app.route("/submit", methods=['GET', 'POST'])
def get_output():
    if request.method == 'POST':
        img = request.files['my_image']
        if img:
            img_path = os.path.join("static", img.filename)
            img.save(img_path)
            pred_Tumer = predict_label(img_path)
            return render_template("project.html", predicted_class=pred_Tumer, img_path=img_path)
        else:
            return render_template("project.html", predicted_class="No image uploaded")
    return render_template("project.html")

@app.route("/TestImages/<fname>", methods=['GET'])
def get_img(fname):
    return send_file(f'TestImages/{fname}')

if __name__ == '__main__':
    app.run(debug=True)
