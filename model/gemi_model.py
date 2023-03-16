from visualprocessing.utils import *
import tensorflow as tf
import numpy as np
import json
import os

'''
Single Frame Tensorflow Classifier for Meteors
'''

class SimpleMeteorClassifier():
    def __init__(self, model_path):
        # read model/gemi_model.json file to read input shape and dimensions
        with open('model/gemi_model.json') as json_file:
            model_properties = json.load(json_file)
        self.dim = (model_properties['height'], model_properties['width'])
        self.class_labels = model_properties['class_labels']
        self.class_labels = {v: k for k, v in self.class_labels.items()}

        self.classifier = tf.keras.models.load_model(model_path, custom_objects=None, compile=True)


    def predict(self, frame, bboxes):
        # Crop detected objects in frame
        pred_labels = []
        if len(bboxes)>0:
            cropped_images = np.array([crop_with_padding(frame, bbox[:-1], self.dim) for bbox in bboxes])
            
            #Predict if Boxes contain Meteors
            pred_labels = self.classifier.predict(cropped_images, verbose=0)
            pred_labels = self.get_labels(pred_labels)
        return pred_labels


    def get_labels(self, preds):
        preds = np.round(preds,0).astype(int)
        predicted_classes = [self.class_labels[int(i)] for i in preds]
        return predicted_classes

        