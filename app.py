##Importing required libraries
import sys
import os
import glob
import numpy as np
import tensorflow as tf

from tensorflow.keras.applications.vgg19 import preprocess_input
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

from flask import *
from werkzeug.utils import secure_filename

##define flask app
app = Flask(__name__)

##We load the saved model by keras
model = load_model("skin_cancer_model_vgg19.h5")

def model_predict(img_path, model):
    print(img_path)
    img =image.load_img(img_path, target_size=(224,224))
    X = image.img_to_array(img)
    #scaling
    X =X/255.0
    X = np.expand_dims(X, axis = 0)

    preds = model.predict(X)
    preds = np.argmax(preds, axis = 1)

    if preds == 0:
        preds = "Great!!! You don't have skin cancer disease"

    else:
        preds = "Sorry, you have skin cancer disease, kindly contact your doctor."

    return preds


@app.route('/', methods = ['GET'])
def index():
    ## Main Page
    return render_template('index.html')

@app.route('/predict', methods = ['GET', 'POST'])
def upload():
    if request.method == 'POST':
        #Get the file from post request
        f = request.files['file']

        #save the files to ./uploads
        basepath =os.path.dirname(__file__)
        file_path =os.path.join(basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)

        ## Make predictions
        preds =model_predict(file_path, model)
        result = preds
        return result
    return None

if __name__ == '__main__':
    app.run(port = 5001, debug= True)
