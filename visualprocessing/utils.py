import cv2
import numpy as np


def scale_img(original_img, scale_percent=100):
    if scale_img == 100:
        return original_img
    
    #width = 720
    height = 720
    #height = original_img.shape[0]
    #scale_percent = height * 100/original_img.shape[0]
    height = int(original_img.shape[0] * scale_percent / 100)
    width = int(original_img.shape[1] * scale_percent / 100)
    dim = (width, height)
    resized_frame = cv2.resize(original_img, dim, interpolation=cv2.INTER_AREA)
    return resized_frame 

def resize_and_pad(img, target_size):
    # Determine the larger dimension
    h, w = img.shape[:2]
    if h > w:
        new_h = target_size[0]
        new_w = int(w * new_h / h)
    else:
        new_w = target_size[1]
        new_h = int(h * new_w / w)

    # Resize the image while maintaining aspect ratio
    resized_img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)

    # Pad the smaller dimension to center the image
    padded_img = np.zeros((target_size[0], target_size[1], 3), dtype=np.uint8)
    start_h = (target_size[0] - new_h) // 2
    start_w = (target_size[1] - new_w) // 2
    padded_img[start_h:start_h+new_h, start_w:start_w+new_w, :] = resized_img

    return padded_img

def crop_with_padding(original_image, boxpoints, img_dim):
    # Define target size
    target_size = img_dim

    # Crop image from original image
    x, y, w, h = boxpoints
    cropped_img = original_image[y:y+h, x:x+w]

    # Resize image to fit target size while maintaining aspect ratio
    resized_img = resize_and_pad(cropped_img, target_size)

    return resized_img

def pointInRect(point, rect):
    x1, y1, w, h = rect
    x2, y2 = x1+w, y1+h
    x, y = point
    if (x1 <= x and x <= x2) and (y1 <= y and y <= y2):
        return True
    return False

def get_xy_from_bbox(bbox):
    x1, y1, w, h = bbox
    x2, y2 = x1+w, y1+h
    return x1, x2, y1, y2

def get_min_max_coords(bbox_list):
    minx, miny, maxx, maxy = float('inf'), float('inf'), float(-1), float(-1)
    for bbox in bbox_list:
        x1, x2, y1, y2 = get_xy_from_bbox(bbox)
        if x1 < minx: minx = x1
        if x2 > maxx: maxx = x2
        if y1 < miny: miny = y1
        if y2 > maxy: maxy = y2
    return minx, miny, maxx, maxy

def get_fixed_box_imgs(frame_list, bbox_list):
    minx, miny, maxx, maxy = get_min_max_coords(bbox_list)
    img_list = []
    for frame in frame_list:
        cropped_img = frame[miny:maxy, minx:maxx]
        img_list.append(cropped_img)
    return img_list


def noise_reduction(original_img):
    '''
    Reduces noise in images, caution processing-intensive!
    Parameter:
    P3 - size of the template patch used to calculate weights in pixels.
    P4 - size of the window in pixels used to calculate a weighted average for the specified pixel.
    P5 - parameter controlling the filter strength for the luminance component.
    P6 - As above, but for color components // Not used in a grayscale image.
    '''

    dst = cv2.fastNlMeansDenoisingColored(original_img, None, 10, 10, 7, 15) 
    return dst

def scale_coordinates(bboxids, scale):
    if scale == 1:
        return bboxids
    else:
        return [int(np.round(i*scale, 0)) for i in bboxids[:-1]] + [bboxids[-1]]

