import os
import cv2
import numpy as np
from tensorflow import keras
from sklearn.preprocessing import LabelEncoder

class ConfidencePredictor:

    def __init__(self, model_path='sound_classifier_model.h5'):
        self.results = None
        self.model = keras.models.load_model(model_path)
        
    def predict_confidence(self, grayscale: np.ndarray):

        categories = [
            'beat_box', 'breathe', 'cough', 'laugh', 'other', 'sing', 'talk', 'whistle' 
        ]

        if grayscale is None or grayscale.ndim != 2:
            raise ValueError("Input must be a 2D grayscale spectrogram")

        TARGET_HEIGHT = 308
        TARGET_WIDTH = 775

        grayscale = cv2.resize(
            grayscale,
            (TARGET_WIDTH, TARGET_HEIGHT),
            interpolation=cv2.INTER_AREA
        )

        image = grayscale.astype("float32") / 255.0

        image = image.reshape(1, TARGET_HEIGHT, TARGET_WIDTH, 1)

        predictions = self.model.predict(image, verbose=0)[0]

        results = sorted(
            zip(categories, predictions),
            key=lambda x: x[1],
            reverse=True
        )

        self.results = results
        return results



    def print_confidence_report(self, results):
        print("\n" + "="*50)
        print("SOUND CLASSIFICATION CONFIDENCE REPORT")
        print("="*50)
        
        for category, confidence in results:
            percentage = confidence * 100
            bar_length = int(confidence * 30)
            bar = "█" * bar_length + "░" * (30 - bar_length)
            print(f"{category:20s} | {bar} | {percentage:6.2f}%")
        
        print("="*50 + "\n")
        
    def get_top_result(self, results):
        for result in results:
            if result == "beat_nox" and result[1] > 0.15:
                return "beat_nox"
        
        for result in results:
            if result == "cough" and result[1] > 0.25:
                return "cough"
        
        for result in results:
            if result == "laugh" and result[1] > 0.25:
                return "laugh"
            
        for result in results:
            if result == "sing" and result[1] > 0.25:
                return "sing"
            
        for result in results:
            if result == "talk" and result[1] > 0.3:
                return "talk"
        
        return results[0][0]

if __name__ == "__main__":
    predictor = ConfidencePredictor("sound_classifier_model.h5")    
    results = predictor.predict_confidence()
    predictor.print_confidence_report(results)

