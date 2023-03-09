from cv2 import threshold
from flask import Flask, render_template, Response, request
import datetime, time
import os
from threading import Thread

import cv2
from visualprocessing.buffer import FrameBuffer
from visualprocessing.detector import ObjectDetector
from visualprocessing.tracker import EuclideanDistTracker
from visualprocessing.utils import scale_img
from generator.image_generator import ImageGenerator
from visualprocessing.classifier import ObjectClassifier

from model.meteor_model import MeteorModel
import numpy as np
import torch


# make snapshots directory to save pics
snapshot_path = './snapshot'
try:
    os.mkdir(snapshot_path)
except OSError as error:
    pass


#instantiate flask app  
app = Flask(__name__)
app.config.from_pyfile('config.py')


# global bool case variables
global onoff, detect, subtract, capture, rec
onoff=True
detect=True
subtract=False
capture=False
rec=False


global camera, rec_frame, out, video_file

global threshold1, ksize1
threshold = 60
ksize1 = 5


# instantiate 
frame_buffer = FrameBuffer(write_out=False)
object_detector = ObjectDetector(method='diff') # diff or subtractKNN
object_tracker = EuclideanDistTracker()
object_classifier = ObjectClassifier(base_path=os.getcwd())

meteor_model = MeteorModel(weithts_path='model/model_spp_crossval_v4.pt')


def start_videosource():
    global camera, video_file
    if input_source=='cam':
        camera = cv2.VideoCapture(video_file)
    elif input_source=='gen':
        camera = ImageGenerator(width=720, height=480)
        camera.start()


def record(out):
    global rec_frame
    while(rec):
        time.sleep(0.05)
        out.write(rec_frame)

def mark_objects(img, box_ids, labels):
    for box_id, label in zip(box_ids, labels):
        x, y, w, h, id = box_id
        cv2.putText(img, str(id), (x,y-15), cv2.FONT_HERSHEY_PLAIN, 1, (255,0,0),2)
        cv2.putText(img, str(label), (x,y+30), cv2.FONT_HERSHEY_PLAIN, 1, (0,0,255),2)
        cv2.rectangle(img, (x+int(w/2)-30, y+int(h/2)-30), (x+int(w/2)+30, y+int(h/2)+30), (0,255,0), 2)
        #cv2.rectangle(img, (x-30, y-30), (x+w+30, y+h+30), (0,255,0), 2)
        

    return img



def detect_objects(frame, frameid, show_thresh=False):

    # Continue if not enough frames in buffer
    if (len(frame_buffer.get_frameids())<2):
        return frame

    
    # Object Detection
    detections, mask = object_detector.detect(frame_buffer.get_frame(frameid))

    if show_thresh:
        mask = cv2.cvtColor(src=mask, code=cv2.COLOR_BGR2RGB)
        return mask


    # Object Tracking
    box_ids = object_tracker.update(detections)
    box_ids = [box_ids[k]+[k] for k in box_ids]


    # Add boxes to framebuffer
    frame_buffer.add_bboxes_with_ids(frameid, box_ids)

    # Predict box labels and show them in frame
    box_labels = object_classifier.predict(frame, box_ids)
    frame = mark_objects(frame.copy(), box_ids=box_ids, labels=box_labels)

    return frame            


def gen_frames(): 
    """generate frame by frame from camera"""
    global out, capture, rec_frame
    while True:
        success, frame = camera.read() 
        if success:

            # Set current frame number as frameid
            frameid = int(camera.get(cv2.CAP_PROP_POS_FRAMES))

            if frameid in frame_buffer.frame_ids:
                continue

            frame = scale_img(frame)
            frame = cv2.cvtColor(src=frame, code=cv2.COLOR_BGR2RGB)

            # Add current frame to framebuffer
            frame_buffer.set_filepath(video_file)
            frame_buffer.add_frame(frameid, frame.copy())

            if detect:
                frame = detect_objects(frame, frameid, show_thresh=False)
                
            elif subtract:
                frame = detect_objects(frame, frameid, show_thresh=True)


            if len(frame_buffer.sum_image_dict) != 0:
                bboxes = frame_buffer.sum_image_dict.keys()
                sum_image_list = list(frame_buffer.sum_image_dict.values())


                #print(sum_image_list)

                #sum_image_list = torch.tensor(sum_image_list).permute(0, 3, 2, 1)

                #print(sum_image_list.shape)
                #sum_image_list = sum_image_list.float()

                #print(sum_image_list)
                
                if len(sum_image_list) != 0:
                    labels = []
                    for img in sum_image_list:
                        img = torch.tensor(img).unsqueeze(0).float()
                        img = img.permute(0, 3, 2, 1)
                        labels.append(meteor_model.predict_label(img))
                    print(labels)

                    #print(y)

                    # highlight bboxes in frame with meteor labels
                    for bbox, label in zip(bboxes, labels):
                        x, y, w, h = bbox
                        #print(bbox)
                        if label=='meteor':
                            #cv2.rectangle(frame, (x-30, y-30), (x+w+30, y+h+30), (255,0,0), 2)
                            #cv2.addText(frame, 'meteor', (x,y-15), cv2.FONT_HERSHEY_PLAIN, 1, (255,0,0),2)

                            # add sum image to frame at bbox position
                            #sum_image = frame_buffer.sum_image_dict[bbox]
                            cv2.putText(frame, 'METEOR', (x,y+30), cv2.FONT_HERSHEY_PLAIN, 1, (0,255,255),2)
                            cv2.rectangle(frame, (x+int(w/2)-30, y+int(h/2)-30), (x+int(w/2)+30, y+int(h/2)+30), (0,255,0), 10)




            # Update framebuffer (delete old frames, save frames with objects)
            frame_buffer.update(buffer_min_size=2)



            if capture:
                capture = False
                now = datetime.datetime.now()
                p = os.path.sep.join([snapshot_path, "shot_{}.png".format(str(now).replace(":",''))])
                cv2.imwrite(p, frame)
            
            if rec:
                rec_frame = frame
                frame = cv2.putText(frame, "Recording...", (0,25), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255),4)
            

            try:
                ret, buffer = cv2.imencode('.jpg', frame)
                frame = buffer.tobytes()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            except Exception as e:
                pass
                
        else:
            pass
        



@app.route('/')
def index():
    """AIVA home page."""
    return render_template('index.html')
    
    
@app.route('/video_feed')
def video_feed():
    """Video streaming route. Put this in the src attribute of an img tag."""
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/requests', methods=['POST', 'GET'])
def tasks():
    global onoff, camera, detect, subtract, frame_buffer, threshold1, ksize1
    if request.method == 'POST':

        if  request.form.get('stop') == 'Stop/Start':     
            onoff = not onoff
            if onoff:
                onoff = not onoff
                start_videosource()
            else:
                camera.release()
                cv2.destroyAllWindows()
                frame_buffer.clear_all()

        elif  request.form.get('detect') == 'Detect':
            detect = not detect                
            if detect:
                subtract = False
                time.sleep(4)
                
        elif  request.form.get('subtract') == 'Subtract':
            subtract = not subtract
            if subtract:
                detect = False
                time.sleep(4)


        elif request.form.get('click') == 'Capture':
            global capture
            capture = True
                
        elif  request.form.get('rec') == 'Start/Stop Recording':
            global rec, out
            rec = not rec
            if rec:
                now=datetime.datetime.now() 
                fourcc = cv2.VideoWriter_fourcc(*'XVID')
                out = cv2.VideoWriter('vid_{}.avi'.format(str(now).replace(":",'')), fourcc, 20.0, (640, 480))
                
                #Start new thread for recording the video
                thread = Thread(target = record, args=[out,])
                thread.start()
            else:
                out.release()
     
     
    elif request.method=='GET':
        return render_template('index.html')
    return render_templcate('index.html')

if __name__ == '__main__':
    video_file = '/mnt/disk1/KILabDaten/Geminiden2021/Kamera2/CutVideos/true_cam2_NINJA3_S001_S001_T001_1.mov'
    input_source = 'cam'    #cam or gen
    start_videosource()
    app.run()
    
camera.release()
cv2.destroyAllWindows()
