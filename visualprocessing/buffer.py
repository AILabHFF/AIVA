
class FrameObject():
    '''
    Class for tracking frames with still active use
    '''
    def __init__(self, frameid, frame):
        self.frame_id = frameid
        self.frame = frame
        self.processed_frame = None
        self.object_ids = []
        self.object_bboxes = {}

    
    def add_processedframe(self, processed_frame):
        self.processed_frame = processed_frame

    def add_object(self, objectid, objectbbox):
        self.object_ids.append(objectid)
        self.object_bboxes[objectid] = objectbbox

    def get_frame_id(self):
        return self.frame_id

    def get_frame(self):
        return self.frame

    def get_processed_frame(self):
        return self.processed_frame

    def get_object_ids(self):
        return self.object_ids

    def get_bbox(self, objectid):
        return self.object_bboxes[objectid]



class FrameBuffer():
    def __init__(self):
        self.frame_ids = []
        self.frame_instances = []


    def add_frame(self, frameid, frame):
        self.frame_ids.append(frameid)
        self.frame_instances.append(FrameObject(frameid, frame))

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