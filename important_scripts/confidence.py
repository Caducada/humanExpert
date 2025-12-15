import os
import cv2
import numpy as np
from tensorflow import keras
from sklearn.preprocessing import LabelEncoder

class ConfidencePredictor:

    def __init__(self, test_image_path, model_path='sound_classifier_model.h5'):
        self.model = keras.models.load_model(model_path)
        self.test_image_path = test_image_path
        

    def get_category_labels(self, spectrograms_dir="spectograms"):
        """
        Get all category labels from the spectrograms directory.
        
        Args:
            spectrograms_dir: Directory containing category subdirectories
        
        Returns:
            categories: List of category names
        """
        categories = []
        for category in os.listdir(spectrograms_dir):
            category_path = os.path.join(spectrograms_dir, category)
            if os.path.isdir(category_path):
                categories.append(category)
        
        return sorted(categories)

    def predict_confidence(self):
        """
        Predict confidence scores for all categories for a given spectrogram image.
        
        Args:
            image_path: Path to the spectrogram image
            model: Loaded Keras model
            categories: List of category names
        
        Returns:
            results: List of tuples (category, confidence_score) sorted by confidence (highest first)
        """
        
        # Load and preprocess the image
        image = cv2.imread(self.test_image_path, cv2.IMREAD_GRAYSCALE)
        
        categories = self.get_category_labels()
        
        if image is None:
            raise ValueError(f"Could not load image from {self.test_image_path}")
        
        # Reshape and normalize
        image = image.reshape(1, image.shape[0], image.shape[1], 1)
        image = image / 255.0
        
        # Make prediction
        predictions = self.model.predict(image, verbose=0)
        confidence_scores = predictions[0]
        
        # Create results with category names and confidence scores
        results = []
        for category, confidence in zip(categories, confidence_scores):
            results.append((category, float(confidence)))
        
        # Sort by confidence score (highest first)
        results.sort(key=lambda x: x[1], reverse=True)
        
        return results

    def print_confidence_report(self, results):
        """
        Print a formatted confidence report.
        
        Args:
            results: List of tuples (category, confidence_score)
        """
        print("\n" + "="*50)
        print("SOUND CLASSIFICATION CONFIDENCE REPORT")
        print("="*50)
        
        for category, confidence in results:
            percentage = confidence * 100
            bar_length = int(confidence * 30)
            bar = "█" * bar_length + "░" * (30 - bar_length)
            print(f"{category:20s} | {bar} | {percentage:6.2f}%")
        
        print("="*50 + "\n")

if __name__ == "__main__":
    predictor = ConfidencePredictor(r"spectograms\sing\2025-12-13+23-06-54_part2.png")       
    results = predictor.predict_confidence()
    predictor.print_confidence_report(results)

