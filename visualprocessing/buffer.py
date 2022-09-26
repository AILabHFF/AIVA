import sys

class FrameObject():
    '''
    Class for tracking frames with still active use
    '''
    def __init__(self, frameid, frame):
        self.frame_id = frameid
        self.frame = frame
        self.box_ids = []
        self.bboxes_with_ids = {}


    def add_bboxes_with_ids(self, bbox_and_ids):
        for box_id in bbox_and_ids:
            x, y, w, h, id = box_id
            self.box_ids.append(id)
            self.bboxes_with_ids[id] = [x,y,w,h]

    def get_frame_id(self):
        return self.frame_id

    def get_frame(self):
        return self.frame

    def get_box_ids(self):
        return self.box_ids

    def get_bbox(self, box_id):
        return self.bboxes_with_ids[box_id]



class FrameBuffer():
    def __init__(self):
        self.current_frame_id = 0
        self.frame_ids = []
        self.frame_instances = []


    def add_frame(self, frameid, frame):
        if frameid in self.frame_ids:
            sys.exit('Image with same ID already in buffer')
        self.current_frame_id = frameid
        self.frame_ids.append(frameid)
        self.frame_instances.append(FrameObject(frameid, frame))

    def add_bboxes_with_ids(self, frameid, boxes_with_ids):
        self.get_frameInstance(frameid).add_bboxes_with_ids(boxes_with_ids)

    def get_frameids(self):
        return self.frame_ids
    
    def get_frame(self, frameid):
        return self.get_frameInstance(frameid).get_frame()


    def get_frameInstance(self, frameid):
        for frameinst in self.frame_instances:
            if frameinst.frame_id == frameid:
                return frameinst

    def del_frameInstance(self, frameid):
        self.frame_instances.remove(self.get_frameInstance(frameid))

    def add_processedFrame(self, frameid, processed_frame):
        self.get_FrameInstance(frameid).add_processedframe(processed_frame)


    def add_object(self, frameid, objectid, objectbbox):
        self.get_FrameInstance(frameid).add_object(objectid, objectbbox)



    # More complex returns
    def get_frameInstanceFromObjectID(self, objectid):
        frames = []
        for frameid in self.frame_ids:
            finst = self.get_frameInstance(frameid)
            if objectid in finst.get_object_ids():
                frames.append(finst.get_frame(frameid))
        return frames

    def remove_from_start_excluding_frameid(self, frameid):
        i = self.frame_ids.index(frameid)
        self.frame_ids = self.frame_list[i:]
        self.frame_instances = self.id_list[i:]


    #TODO check below vvv

    def update(self, current_frameid, active_objects, n_buffer=10):
        if len(self.get_frameids())<n_buffer:
            n_buffer = len(self.get_frameids())
            
        for fid in self.frame_ids.copy():
            if fid < current_frameid-n_buffer:
                self.remove_frame(fid)



    def remove_object(self, objectid):
        pass

    def remove_frame(self, frameid):
        self.del_frameInstance(frameid)
        self.frame_ids.remove(frameid)


    def clear_all(self):
        self.frame_ids = []
        self.frame_instances = []