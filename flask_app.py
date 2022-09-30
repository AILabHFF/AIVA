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


global capture, rec_frame, grey, switch, detect, rec, out, image_gen
capture=0
grey=0
detect=1
switch=1
rec=0

#make shots directory to save pics
try:
    os.mkdir('./shots')
except OSError as error:
    pass


#instatiate flask app  
app = Flask(__name__, template_folder='./templates', static_folder='./static')

video_file = '/media/disk1/KILabDaten/Geminiden 2021/Kamera2/CutVideos/true_cam2_NINJA3_S001_S001_T001_1.mov'


frame_buffer = FrameBuffer(video_file)
object_detector = ObjectDetector()
object_tracker = EuclideanDistTracker()

input_source = 'gen'
if input_source=='cam':
    camera = cv2.VideoCapture(video_file)
else:
    camera = ImageGenerator(width=720, height=480)



def record(out):
    global rec_frame
    while(rec):
        time.sleep(0.05)
        out.write(rec_frame)

def mark_objects(img, box_ids):
    for box_id in box_ids:
        x, y, w, h, id = box_id
        cv2.putText(img, str(id), (x,y-15), cv2.FONT_HERSHEY_PLAIN, 1, (255,0,0),2)
        cv2.rectangle(img, (x, y), (x+w, y+h), (0,255,0), 3)
    return img

def detect_objects(frame, frameid):
        img_rgb = cv2.cvtColor(src=frame, code=cv2.COLOR_BGR2RGB)
        img = img_rgb.copy()

        # Add current frame to framebuffer
        frame_buffer.add_frame(frameid, img)

        # Continue if not enough frames in buffer
        if (len(frame_buffer.get_frameids())<2):
            return frame
        
        # Object Detection
        detections = object_detector.detect(img_rgb)
        # Object Tracking
        box_ids = object_tracker.update(detections)

        # Add boxes to framebuffer
        frame_buffer.add_bboxes_with_ids(frameid, box_ids)

        frame = mark_objects(frame, box_ids)

        # Update framebuffer: delete old frames, clean objects etc.
        frame_buffer.update(n_buffer=2)
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

def gen_frames():  # generate frame by frame from camera
    global out, capture, rec_frame
    while True:
        success, frame = camera.read() 
        if success:

            # Set current frame number as frameid
            frameid = int(camera.get(cv2.CAP_PROP_POS_FRAMES))

            frame = scale_img(frame)

            if(detect):                
                frame= detect_objects(frame, frameid)
            if(grey):
                frame = cv2.bitwise_not(frame)   
            if(capture):
                capture=0
                now = datetime.datetime.now()
                p = os.path.sep.join(['shots', "shot_{}.png".format(str(now).replace(":",''))])
                cv2.imwrite(p, frame)
            
            if(rec):
                rec_frame=frame
                frame= cv2.putText(frame, "Recording...", (0,25), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255),4)
            
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
    return render_template('index.html')
    
    
@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/requests',methods=['POST','GET'])
def tasks():
    global switch, camera
    if request.method == 'POST':
        if request.form.get('click') == 'Capture':
            global capture
            capture=1
        elif  request.form.get('grey') == 'Grey':
            global grey
            grey=not grey
        elif  request.form.get('detect') == 'Detect':
            global detect
            detect=not detect 
            if(detect):
                time.sleep(4)   
        elif  request.form.get('stop') == 'Stop/Start':
            
            if(switch==1):
                switch=0
                camera.release()
                cv2.destroyAllWindows()
                
            else:
                if input_source=='cam':
                    camera = cv2.VideoCapture(video_file)
                else:
                    camera = ImageGenerator(width=720, height=480)
                switch=1
        elif  request.form.get('rec') == 'Start/Stop Recording':
            global rec, out
            rec= not rec
            if(rec):
                now=datetime.datetime.now() 
                fourcc = cv2.VideoWriter_fourcc(*'XVID')
                out = cv2.VideoWriter('vid_{}.avi'.format(str(now).replace(":",'')), fourcc, 20.0, (640, 480))
                #Start new thread for recording the video
                thread = Thread(target = record, args=[out,])
                thread.start()
            elif(rec==False):
                out.release()
                          
                 
    elif request.method=='GET':
        return render_template('index.html')
    return render_template('index.html')


if __name__ == '__main__':
    app.run()
    
camera.release()
cv2.destroyAllWindows()     