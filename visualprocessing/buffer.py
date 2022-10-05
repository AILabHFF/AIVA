import sys, os
import cv2
from visualprocessing.utils import *

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
    def __init__(self, write_out=False):
        self.current_frame_id = 0
        self.frame_ids = []
        self.frame_instances = []
        self.capture = write_out

    def set_filepath(self, file_path):
        self.file_path = file_path


    def add_frame(self, frameid, frame):
        if frameid in self.frame_ids:
            print('Image with same ID already in buffer')
            #sys.exit('Image with same ID already in buffer')
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
        



    def update(self, n_buffer=2):
        removed_boxids = []


        keep_in_buffer = self.frame_ids[-n_buffer:]
        dont_keep = self.frame_ids[:-n_buffer]

        # active boxids that are in at least one frame that remains in buffer
        active_ids_in_buffer = set()
        for fid in keep_in_buffer:
            active_ids_in_buffer.update(self.get_frameInstance(fid).get_box_ids())

        # clean frameids from frames with in buffer active boxids
        for fid in dont_keep.copy():
            if any(x in self.get_frameInstance(fid).get_box_ids() for x in active_ids_in_buffer):
                dont_keep.remove(fid)

        # get final boxids to remove
        for fid in dont_keep:
            frame_boxids = self.get_frameInstance(fid).get_box_ids()
            if len(frame_boxids)>0:
                removed_boxids += frame_boxids

        # write images for removed frames
        if self.capture:
            for boxid in set(removed_boxids):
                self.save_as_png(boxid)
        
        # Finally remove all frames without active objects and not in buffer
        for fid in dont_keep:
            self.remove_frame(fid)


    def clear_all(self):

        # active boxids that are in at least one frame that remains in buffer
        active_ids_in_buffer = set()
        for fid in self.frame_ids:
            active_ids_in_buffer.update(self.get_frameInstance(fid).get_box_ids())

        # write images for removed frames
        if self.capture:
            for boxid in active_ids_in_buffer:
                self.save_as_png(boxid)


        self.current_frame_id = 0
        self.frame_ids = []
        self.frame_instances = []


    def remove_frame(self, frameid):
        self.del_frameInstance(frameid)
        self.frame_ids.remove(frameid)



    def get_frames_for_id(self, objectid):
        frames = []
        fids = []
        bboxes = []
        for frameid in self.frame_ids:
            finst = self.get_frameInstance(frameid)
            if objectid in finst.get_box_ids():
                frames.append(self.get_frame(frameid))
                fids.append(frameid)
                bboxes.append(finst.get_bbox(objectid))
        return frames, fids, bboxes


    def save_as_png(self, objectid):
        file_name = os.path.basename(os.path.realpath(self.file_path))
        img_path = './data/saved/'
        dir = img_path+file_name + '/' + str(objectid) + '/'
        if not os.path.isdir(dir):
            os.makedirs(dir)
            print("Directory '%s' created" %dir)

        #save full frame sequence
        if not os.path.isdir(dir+'full_frames'):
            os.makedirs(dir+'full_frames/')

        frames, frameids, bboxes = self.get_frames_for_id(objectid)
        for fid, frame in zip(frameids, frames):
            filedir = dir+'full_frames/' + str(fid) + '.png'
            cv2.imwrite(filedir, frame)


        #save cropped full movement area image
        if not os.path.isdir(dir+'croppped_frames'):
            os.makedirs(dir+'croppped_frames/')

        img_list = get_fixed_box_imgs(frames, bboxes)
        #print(objectid)
        for fid, frame in zip(frameids, img_list):
            filedir = dir+'croppped_frames/' + str(fid) + '.png'
            cv2.imwrite(filedir, frame)

        #save cropped and centered
        

        #save sequence in one image
