import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import os
import numpy as np

class PredictionPipeline:
    def __init__(self, filename):
        self.filename = filename
    
    def predict(self, ):
        model = load_model(os.path.join("artifacts","training", "model.keras"))
        imagename = self.filename
        test_image = image.load_img(imagename, target_size=(224, 224))
        test_image = image.img_to_array(test_image)
        test_image = np.expand_dims(test_image, axis=0)

        result_raw = model.predict(test_image)
        result = np.argmax(result_raw[0], axis=0)
        print(result_raw)
        print(result)

        if result == 0:
            prediction = "Cyst"
            return (prediction, result_raw)
        elif result ==  1:
            prediction = "Normal"
            return (prediction, result_raw)
        elif result == 2:
            prediction = "Stone"
            return (prediction, result_raw)
        elif result == 3:
            prediction = "Tumor"
            return (prediction, result_raw)
        
        return ("Failed", result_raw)
