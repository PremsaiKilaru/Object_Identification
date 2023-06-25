from flask import Flask , render_template,url_for,redirect,request
from werkzeug.utils import secure_filename
import os

from keras.applications.imagenet_utils import preprocess_input, decode_predictions
from keras.utils import load_img, img_to_array
import numpy as np

app = Flask(__name__)


from keras.applications import ResNet50
model = ResNet50(weights='imagenet')

def modelPredict(imgPath,model):
    image1 = load_img(imgPath, target_size = (224, 224))
    transformed_image = img_to_array(image1)
    transformed_image = np.expand_dims(transformed_image, axis = 0)
    transformed_image = preprocess_input(transformed_image,mode='caffe')
    prediction = model.predict(transformed_image)
    predictionLabel = decode_predictions(prediction, top = 3)
    return predictionLabel

@app.route('/',methods=['GET'])
def Home():
    return render_template('home.html')

@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        # Get the file from post request
        f = request.files['file']

        # Save the file to ./uploads
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)

        # Make prediction
        preds = modelPredict(file_path, model)

   # ImageNet Decode
        result = str(preds[0][0][1]) + " Accuracy : " +  str(preds[0][0][2]*100)           # Convert to string
        return result
    return None     


if __name__== "__main__":
    app.run(debug = True)