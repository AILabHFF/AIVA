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

# instantiate 
frame_buffer = FrameBuffer()
object_detector = ObjectDetector(method='diff') # diff or subtractKNN
object_tracker = EuclideanDistTracker()
object_classifier = ObjectClassifier()



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
        #cv2.rectangle(img, (x+int(w/2)-30, y+int(h/2)-30), (x+int(w/2)+30, y+int(h/2)+30), (0,255,0), 2)
        cv2.rectangle(img, (x-30, y-30), (x+w+30, y+h+30), (0,255,0), 2)
        

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



def scale_img(original_img, scale_percent=100):
    if scale_img == 100:
        return original_img
    
    #width = 720
    height = 720
    
    scale_percent = 720 * 100/original_img.shape[0]

    width = int(original_img.shape[1] * scale_percent / 100)
    dim = (width, height)
    resized_frame = cv2.resize(original_img, dim, interpolation=cv2.INTER_AREA)
    return resized_frame 

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
    global onoff, camera, detect, subtract, frame_buffer
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
    return render_template('index.html')

if __name__ == '__main__':
    video_file = '/media/disk1/KILabDaten/Geminiden 2021/Kamera2/CutVideos/true_cam2_NINJA3_S001_S001_T001_1.mov'
    input_source = 'gen'
    start_videosource()
    app.run()
    
camera.release()
cv2.destroyAllWindows()
