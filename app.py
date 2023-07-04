import os
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS, cross_origin
from Crop_Image_Classification.utils.common import decodeImage
from Crop_Image_Classification.pipeline.predict import PredictionPipeline

os.putenv('LANG', 'en_US.UTF-8')
os.putenv('LC_ALL', 'en_US.UTF-8')

app = Flask(__name__)
CORS(app)


class App:
    def __init__(self):
        self.filename = "inputImage.jpg"
        self.classifier = PredictionPipeline(self.filename)


@app.route("/", methods=['GET'])
@cross_origin()
def home():
    return render_template('index.html')


@app.route("/train", methods=['GET', 'POST'])
@cross_origin()
def trainRoute():
    os.system("python main.py")
    return "Training done successfully!"


@app.route("/predict", methods=['POST'])
@cross_origin()
def predictRoute():
    image = request.json['image']
    decodeImage(image, clApp.filename)
    confidence_interval, class_name = clApp.classifier.predict()

    result = {
        'confidence_interval': str(confidence_interval),
        'class_name': str(class_name)
    }
    print(result)
    return jsonify([result])


if __name__ == "__main__":
    clApp = App()
    app.run(host='0.0.0.0', port=7000, debug=True)  # local host
