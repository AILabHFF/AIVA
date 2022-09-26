#import warnings
#warnings.filterwarnings("ignore", message=r"Passing", category=FutureWarning)

import cv2
from visualprocessing.buffer import FrameBuffer
from visualprocessing.detector import ObjectDetector
from visualprocessing.tracker import EuclideanDistTracker
from visualprocessing.utils import scale_img
from view.view import View
from generator.image_generator import ImageGenerator



def run_video(file_path):
    
    # Load video from file
    cap = cv2.VideoCapture(file_path)

    # Load FrameBuffer
    frame_buffer = FrameBuffer()
    object_detector = ObjectDetector()
    object_tracker = EuclideanDistTracker()
    viewer = View()
    
    imgen = ImageGenerator()

    # Infinite loop until we hit the escape key on keyboard

    frameid = 0
    while True:
        frameid += 1
        # Convert frame to rgb
        _, img_rgb = imgen.next_image()
        img_rgb = cv2.cvtColor(src=img_rgb, code=cv2.COLOR_BGR2RGB)

        # Scaling for processing time and visualization optimization
        #img_rgb = scale_img(img_rgb, 25)

        # Add current frame to framebuffer
        frame_buffer.add_frame(frameid, img_rgb)

        # Continue if not enough frames in buffer
        if (len(frame_buffer.get_frameids())<2):
            continue

        # Object Detection
        detections = object_detector.detect(img_rgb)
        # Object Tracking
        box_ids = object_tracker.update(detections)

        # View
        viewer.update_view(img_rgb)
        viewer.mark_objects(box_ids)
        viewer.view_video()

        # Update framebuffer: delete old frames, clean objects etc. 
        frame_buffer.update(current_frameid=frameid, active_objects=[])

        # If escape button is pressed exit
        k = cv2.waitKey(1)
        if k == 27:
            break





if __name__ == "__main__":
    file_path = '/media/disk1/KILabDaten/Geminiden 2021/Kamera2/CutVideos/true_cam2_NINJA3_S001_S001_T001_1.mov'
    run_video(file_path)
    print('done')