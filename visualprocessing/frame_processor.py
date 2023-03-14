import cv2
from visualprocessing.buffer import FrameBuffer
from visualprocessing.detector import ObjectDetector
from visualprocessing.tracker import EuclideanDistTracker
from visualprocessing.utils import scale_img

import numpy as np
import torch

from model.single_frame_model import SimpleMeteorClassifier
from model.meteor_model import MeteorModel

class VisualProcessor:
    def __init__(self, scale, method, threshold,
                 ksize, object_minsize, object_maxsize, filename):
        
        self.scale = scale
        self.method = method
        self.threshold = threshold
        self.ksize = ksize
        self.object_minsize = object_minsize
        self.object_maxsize = object_maxsize


        # instantiate 
        self.frame_buffer = FrameBuffer(write_out=True)
        self.object_detector = ObjectDetector(method=self.method, minobjectsize=self.object_minsize,
                                              maxobjectsize=self.object_maxsize, threshold=self.threshold, ksize=self.ksize)
        #self.object_detector = ObjectDetector()
        self.object_tracker = EuclideanDistTracker()
        self.frame_classifier = SimpleMeteorClassifier(model_path='model/gemi_model.hdf5')
        self.meteor_model = MeteorModel(weithts_path='model/model_spp_crossval_v4.pt')
        self.filename = filename


    def mark_objects(self, img, box_ids, labels):
        for box_id, label in zip(box_ids, labels):
            x, y, w, h, id = box_id
            cv2.putText(img, str(id), (x,y-15), cv2.FONT_HERSHEY_PLAIN, 1, (255,0,0),2)
            cv2.putText(img, str(label), (x,y+30), cv2.FONT_HERSHEY_PLAIN, 1, (0,0,255),2)
            cv2.rectangle(img, (x+int(w/2)-30, y+int(h/2)-30), (x+int(w/2)+30, y+int(h/2)+30), (0,255,0), 2)
            
        return img

    def detect_objects(self, frame, frameid):

        # Continue if not enough frames in buffer
        if (len(self.frame_buffer.get_frameids())<2):
            return frame

        # Object Detection
        detections, mask = self.object_detector.detect(self.frame_buffer.get_frame(frameid))

        # Object Tracking
        box_ids = self.object_tracker.update(detections)
        box_ids = [box_ids[k]+[k] for k in box_ids]

        # Add boxes to framebuffer
        self.frame_buffer.add_bboxes_with_ids(frameid, box_ids)

        # Predict box labels and show them in frame
        box_labels = self.frame_classifier.predict(frame, box_ids)
        frame = self.mark_objects(frame.copy(), box_ids=box_ids, labels=box_labels)

        return frame        



    def process_frame(self, frame, current_frame_num):
        frameid = current_frame_num

        frame = scale_img(frame, self.scale)

        frame = cv2.cvtColor(src=frame, code=cv2.COLOR_BGR2RGB)


        # Add current frame to framebuffer
        self.frame_buffer.set_filepath(self.filename)
        self.frame_buffer.add_frame(frameid, frame.copy())


        frame = self.detect_objects(frame, frameid)



        if len(self.frame_buffer.sum_image_dict) > 0:
            bboxes = self.frame_buffer.sum_image_dict.keys()
            sum_image_list = list(self.frame_buffer.sum_image_dict.values())

            if len(sum_image_list) > 0:
                labels = []
                for img in sum_image_list:
                    img = torch.tensor(img).unsqueeze(0).float()
                    img = img.permute(0, 3, 2, 1)
                    labels.append(self.meteor_model.predict_label(img))
                print(labels)

                # highlight bboxes in frame with meteor labels
                for bbox, label in zip(bboxes, labels):
                    x, y, w, h = bbox
                    if label=='meteor':
                        # add sum image to frame at bbox position
                        #sum_image = frame_buffer.sum_image_dict[bbox]
                        cv2.putText(frame, 'METEOR', (x,y+30), cv2.FONT_HERSHEY_PLAIN, 1, (0,255,255),2)
                        cv2.rectangle(frame, (x+int(w/2)-30, y+int(h/2)-30), (x+int(w/2)+30, y+int(h/2)+30), (0,255,0), 10)




        # Update framebuffer (delete old frames, save frames with objects)
        self.frame_buffer.update(buffer_min_size=2)

        return frame


