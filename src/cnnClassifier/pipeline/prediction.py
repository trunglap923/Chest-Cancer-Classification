import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from cnnClassifier import logger
import os

class PredictionPipeline:
    def __init__(self,filename):
        self.filename =filename


    def predict(self):
        # load model
        model = load_model(os.path.join("artifacts","training", "model.h5"))

        imagename = self.filename
        test_image = image.load_img(imagename, target_size = (224,224))
        test_image = image.img_to_array(test_image)
        test_image = np.expand_dims(test_image, axis = 0)
        test_image = test_image / 255.0 

        prediction_prob = model.predict(test_image, verbose=0)
        predicted_class_index = np.argmax(prediction_prob, axis=1)[0]
        confidence_score = prediction_prob[0][predicted_class_index]
        logger.info(confidence_score)

        if predicted_class_index == 1:
            prediction = 'Normal'
            logger.info("Prediction is Normal")
            return [{"image" : prediction, "confidence": float(confidence_score)}]
        else:
            prediction = 'Adenocarcinoma Cancer'
            logger.info("Prediction is Adenocarcinoma Cancer")
            return [{"image" : prediction, "confidence": float(confidence_score)}]
