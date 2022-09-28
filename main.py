import cv2
from visualprocessing.buffer import FrameBuffer
from visualprocessing.detector import ObjectDetector
from visualprocessing.tracker import EuclideanDistTracker
from visualprocessing.utils import scale_img, scale_coordinates
from view.view import View


def run_video(file_path, show=True):
    
    # Load video from file
    cap = cv2.VideoCapture(file_path)

    # Load FrameBuffer
    frame_buffer = FrameBuffer(file_path)
    object_detector = ObjectDetector()
    object_tracker = EuclideanDistTracker()
    viewer = View()

    # Scale image for processing bur scale it afterwards back for visualization and writing pngs
    scale = 0.25
    scaleback = 10/scale/10
    
    # Infinite loop until we hit the escape key on keyboard
    while cap.isOpened():
        ret, frame = cap.read()

        if ret == True:
            # Convert frame to rgb
            img_rgb = cv2.cvtColor(src=frame, code=cv2.COLOR_BGR2RGB)
            img = img_rgb.copy()

            # Scaling for processing time and visualization optimization
            img_rgb = scale_img(img_rgb, scale*100)

            # Set current frame number as frameid
            frameid = int(cap.get(cv2.CAP_PROP_POS_FRAMES))

            # Add current frame to framebuffer
            frame_buffer.add_frame(frameid, img)

            # Continue if not enough frames in buffer
            if (len(frame_buffer.get_frameids())<2):
                continue
            
            # Object Detection
            detections = object_detector.detect(img_rgb)
            # Object Tracking
            box_ids = object_tracker.update(detections)

            # Scaled up coordinates
            scaled_boxids = [scale_coordinates(bid, scaleback) for bid in box_ids]

            # Add boxes to framebuffer
            frame_buffer.add_bboxes_with_ids(frameid, scaled_boxids)

            if show:
                # View
                viewer.update_view(img.copy())
                viewer.mark_objects(scaled_boxids)
                viewer.view_video()

            # Update framebuffer: delete old frames, clean objects etc.
            frame_buffer.update(n_buffer=2)

            # If escape button is pressed exit
            k = cv2.waitKey(1)
            if k == 27:
                break
        
        # Break if no frame returned
        else:
            break


    cv2.destroyAllWindows()
    cap.release()





if __name__ == "__main__":
    file_path = '/media/disk1/KILabDaten/Geminiden 2021/Kamera2/CutVideos/true_cam2_NINJA3_S001_S001_T001_1.mov'
    run_video(file_path)
    print('done')
