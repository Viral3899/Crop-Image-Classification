import os
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np


class PredictionPipeline:
    def __init__(self, filename):
        self.filename = filename

    def predict(self):
        # Load model
        model = load_model(os.path.join("artifacts", "training", "model.h5"))

        imagename = self.filename
        test_image = image.load_img(imagename, target_size=(299, 299))  # Update target_size
        test_image = image.img_to_array(test_image)
        test_image = test_image / 255.0  # Normalize the image

        # Expand dimensions to match the input shape expected by the model
        test_image = np.expand_dims(test_image, axis=0)

        # Make the prediction
        predictions = model.predict(test_image)
        predicted_class_index = np.argmax(predictions)

        class_labels = sorted(os.listdir('artifacts\data_ingestion\Agricultural-crops'))  # Replace with the actual path

        prediction = class_labels[predicted_class_index]

        return [{"image": prediction}]
