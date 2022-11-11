import cv2

class View():
    def __init__(self):
        self.img = []

    def view_video(self):
        cv2.imshow('AIVA', self.img)

    def update_view(self, img_rgb):
        self.img = img_rgb

    def mark_objects(self, box_ids):
        for box_id in box_ids:
            x, y, w, h, id = box_id
            cv2.putText(self.img, str(id), (x,y-15), cv2.FONT_HERSHEY_PLAIN, 1, (255,0,0),2)
            cv2.rectangle(self.img, (x, y), (x+w, y+h), (0,255,0), 3)

    