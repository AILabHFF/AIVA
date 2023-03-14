from visualprocessing.utils import *
import numpy as np

class ObjectDetector():
    def __init__(self, method='frameDiff', minobjectsize=10, maxobjectsize=1000, threshold=60, ksize=5):
        self.method = method

        if self.method == 'subtractKNN':
            self.background_subtractor = cv2.createBackgroundSubtractorKNN(history=10)
            #self.background_subtracktor = cv2.createBackgroundSubtractorMOG2(history=10, varThreshold=40)

        self.last_greyblur_frame = []
        self.minobjectsize = minobjectsize
        self.maxobjectsize = maxobjectsize
        self.subtract_threshold = threshold
        self.kernel_size = ksize


    def get_greyblur_frame(self, original_img):
        # Grayscale
        prepared_frame = cv2.cvtColor(original_img, cv2.COLOR_BGR2GRAY)
        # Blur image
        prepared_frame = cv2.GaussianBlur(src=prepared_frame, ksize=(self.kernel_size, self.kernel_size), sigmaX=0)
        return prepared_frame

    def subtractKNN_method(self, original_img):
        mask = self.background_subtractor.apply(original_img)
        _, mask = cv2.threshold(mask, 254, 255, cv2.THRESH_BINARY)
        return mask

    def diff_method(self, greyblur_frame):

        # Calculate difference in frames
        diff_frame = cv2.absdiff(src1=self.last_greyblur_frame, src2=greyblur_frame)
        # Set current frame as last frame for next iteration
        self.last_greyblur_frame = greyblur_frame
        # Dilute the image a bit to make differences more seeable; more suitable for contour detection
        kernel = np.ones((5, 5))
        dillute_frame = cv2.dilate(diff_frame, kernel, 1)
        # Only take different areas that are different enough (>20 / 255)
        thresh_frame = cv2.threshold(src=dillute_frame, thresh=self.subtract_threshold, maxval=255, type=cv2.THRESH_BINARY)[1]
        return thresh_frame


    def detect(self, currentframe):
        '''
        Method for detecting moving objects in frame
        '''
        frame = currentframe

        # 2. Prepare image; grayscale and blur
        current_greyblurframe = self.get_greyblur_frame(frame)

        if self.last_greyblur_frame == []:
            self.last_greyblur_frame = current_greyblurframe
            return [], current_greyblurframe

        if self.method == 'frameDiff':
            mask = self.diff_method(current_greyblurframe)

        if self.method == 'subtractKNN':
            mask = self.subtractKNN_method(frame)

        # Different modes:
        # CV_RETR_EXTERNAL - gives "outer" contours, so if you have (say) one contour enclosing another (like concentric circles), only the outermost is given.
        # CV_RETR_TREE     - calculates the full hierarchy of the contours. So you can say that object1 is nested 4 levels deep within object2 and object3 is also nested 4 levels deep.
        contours, _ = cv2.findContours(image=mask, mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_SIMPLE) # [-2:]

        detections = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > self.minobjectsize and area < self.maxobjectsize:
                x, y, w, h = cv2.boundingRect(contour)
                detections.append([x,y,w,h])

        return detections, mask

