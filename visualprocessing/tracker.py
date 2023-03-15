'''
Simple centroid tracking found and modified from:
https://stackoverflow.com/questions/68491404/kalman-filter-with-python-in-function-euclideandisttracker
https://pyimagesearch.com/2018/07/23/simple-object-tracking-with-opencv/

'''

import math

class EuclideanDistTracker:
    def __init__(self):
        # Store the center positions of the objects
        self.center_points = {}
        # Keep the count of the IDs
        # each time a new object id detected, the count will increase by one
        self.id_count = 1
        self.maxdist = 30

    def update(self, objects_rect):
        # Objects boxes and ids
        objects_bbs_ids_dict = {}

        # Get center point of new object
        for rect in objects_rect:
            x, y, w, h = rect
            cx = (x + x + w) // 2
            cy = (y + y + h) // 2

            # Find out if that object was detected already
            same_object_detected = False
            for id, pt in self.center_points.items():
                dist = math.hypot(cx - pt[0], cy - pt[1])

                if dist < self.maxdist:
                    self.center_points[id] = (cx, cy)
                    objects_bbs_ids_dict[id] = [x, y, w, h]
                    same_object_detected = True
                    break

            # New object is detected we assign the ID to that object
            if same_object_detected is False:
                self.center_points[self.id_count] = (cx, cy)
                objects_bbs_ids_dict[self.id_count] = [x, y, w, h]
                self.id_count += 1

        # Clean the dictionary by center points to remove IDS not used anymore
        new_center_points = {}

        for object_id in objects_bbs_ids_dict:
            center = self.center_points[object_id]
            new_center_points[object_id] = center

        # Update dictionary with IDs not used removed
        self.center_points = new_center_points.copy()
        return objects_bbs_ids_dict
