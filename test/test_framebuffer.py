import unittest
from visualprocessing.buffer import FrameBuffer
import numpy as np
np.random.seed(0)

class TestFrameBuffer(unittest.TestCase):
    '''
    Unittests for framebuffer
    '''

    def setUp(self):
        self.framebuffer = FrameBuffer('/test/test.mp4')
        self.sample_rgbframe = np.random.randint(255, size=(500,500,3),dtype=np.uint8)

    def test_loadbuffer(self):
        framebuffer = FrameBuffer('/test/test.mp4')
        self.assertIsInstance(framebuffer, object)

    def test_add_frame(self):
        id = 1
        frame = self.sample_rgbframe
        self.framebuffer.add_frame(id, frame)        
        self.assertTrue(np.alltrue(self.framebuffer.get_frame(id) == frame))

    def test_del_frame(self):
        for i in range(0,5):
            id = i
            frame = self.sample_rgbframe
            self.framebuffer.add_frame(id, frame)
        self.assertEqual(self.framebuffer.get_frameids(), [0,1,2,3,4])
        self.framebuffer.remove_frame(4)

        #Check that element get gets removed from indexlist
        self.assertEqual(self.framebuffer.get_frameids(), [0,1,2,3])
        #Check that element get gets removed from instancelist
        self.assertEqual([i.get_frame_id() for i in self.framebuffer.frame_instances], [0,1,2,3])

    def test_update(self):
        
        # 1. Test short buffer sequence
        for i in range(0,5):
            id = i
            frame = self.sample_rgbframe
            self.framebuffer.add_frame(id, frame)
        self.assertEqual(self.framebuffer.get_frameids(), [0,1,2,3,4])
        self.framebuffer.update(buffer_min_size=10)
        self.assertEqual(self.framebuffer.get_frameids(), [0,1,2,3,4])


        # 2. Test longer buffer sequence
        self.framebuffer.clear_all()
        for i in range(5,20):
            id = i
            frame = self.sample_rgbframe
            self.framebuffer.add_frame(id, frame)

        self.assertEqual(self.framebuffer.get_frameids(), [i for i in range(5, 20)])
        self.framebuffer.update(buffer_min_size=10)
        self.assertEqual(self.framebuffer.get_frameids(), [i for i in range(10, 20)])


    def test_frame_id(self):
        pass


if __name__ == '__main__':
    unittest.main()