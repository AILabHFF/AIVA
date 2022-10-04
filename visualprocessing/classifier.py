from visualprocessing.utils import *
import tensorflow as tf
import numpy as np

class ObjectClassifier():
    def __init__(self):
        self.dim = (96,96)
        self.class_labels = {0:'???', 1:'meteor'}
        model_path = 'data/gemi_model.hdf5'
        self.classifier = tf.keras.models.load_model(model_path, custom_objects=None, compile=True)


    def predict(self, frame, bboxes):
        # Crop detected objects in frame
        pred_labels = []
        if len(bboxes)>0:
            cropped_images = np.array([crop_with_padding(frame, bbox[:-1], self.dim) for bbox in bboxes])
            # Predict if Boxes contain Meteors
            pred_labels = self.classifier.predict(cropped_images, verbose=0)
            pred_labels = self.get_labels(pred_labels)
        return pred_labels

    def get_labels(self, preds):
        preds = np.round(preds,0).astype(int)
        predicted_classes = [self.class_labels[int(i)] for i in preds]
        return predicted_classes

    def add_padding(self, frame, bbox):
        x, y, w, h, _ = bbox
        # Crop Box for AI and writing png
        return crop_with_padding(frame, (x, y, w, h), self.dim)
        