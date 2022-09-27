## Depth based anti-spoof

## author: mzh, 11/05/2020, ULR
from pdb import set_trace as bp

import torch

import os
import pandas as pd
import cv2
import math
import shutil
import matplotlib.pyplot as plt
import glob

os.environ['CUDA_VISIBLE_DEVICES'] = '0,2'
os.system('echo $CUDA_VISIBLE_DEVICES')

import numpy as np
# from Encoding import load_feature
#import imgaug.augmenters as iaa
from PIL import Image
import statistics
import time
# import sys
# sys.path.append('/home/zming/dlib/dlib-19.20.0')
import dlib

# video_path = '/data/zming/datasets/Anti-spoof/SELFCOLLECT_Data/HuaweiMetaLive/VID_20200622_165521'
# video_path = '/data/zming/datasets/Anti-spoof/SELFCOLLECT_Data/AriadNext/exemple_video_attaque_avec_flash'
# video_path = '/data/zming/datasets/Anti-spoof/SELFCOLLECT_Data/test'
#video_path = '/data/zming/datasets/Anti-spoof/SiW-M/SiW-M-images'
video_path = '/data/zming/datasets/Anti-spoof/SELFCOLLECT_Data/AriadNext/exemple_video_attaque_avec_flash/'

#scale_size = 1.0
scale_size = 0.25

face_scale = 1.0


def rescale_face(image, bbox, scale):
    # f=open(face_name_full,'r')
    # lines=f.readlines()
    # y1,x1,w,h=[float(ele) for ele in lines[:4]]
    # f.close()

    # strbbox=str.split(bbox, ' ')
    # y1=int(strbbox[1])
    # x1=int(strbbox[0])
    # y2=int(strbbox[3])
    # x2=int(strbbox[2])
    # region = image[y1:y2, x1:x2]
    # cv2.imwrite('oringial0.jpg', region)

    y1 = int(bbox[0])
    x1 = int(bbox[1])
    y2 = int(bbox[2])
    x2 = int(bbox[3])
    # region = image[y1:y2, x1:x2]
    # cv2.imwrite('oringial0.jpg', region)

    w = x2 - x1
    h = y2 - y1

    y_mid = ((y1 + y2) / 2.0) / scale
    x_mid = ((x1 + x2) / 2.0) / scale
    h_img, w_img = image.shape[0], image.shape[1]
    # w_img,h_img=image.size
    w_scale = w / scale
    h_scale = h / scale
    y1 = y_mid - h_scale / 2.0
    x1 = x_mid - w_scale / 2.0
    y2 = y_mid + h_scale / 2.0
    x2 = x_mid + w_scale / 2.0
    y1 = max(math.floor(y1), 0)
    x1 = max(math.floor(x1), 0)
    y2 = min(math.floor(y2), h_img)
    x2 = min(math.floor(x2), w_img)
    bbox = [y1, x1, y2, x2]
    region = image[y1:y2, x1:x2]
    return region, bbox


def face_dectection(image_path, vid_len, detector):
    idxs = [i for i, c in enumerate(image_path) if c == '/']
    log_dir = image_path[:idxs[-1]]
    image_name = image_path[idxs[-1] + 1:-4]
    # bp()
    # print('%s'%log_dir)
    f = open(os.path.join(log_dir, '%s_bbox.txt' % image_name), 'w')

    frame = cv2.imread(image_path)

    tt0 = time.time()

    img_size = frame.shape[0:2]
    im_np_scale = cv2.resize(frame, (int(img_size[1] * scale_size), int(img_size[0] * scale_size)),
                             interpolation=cv2.INTER_LINEAR)
    t0 = time.time()
    faces = detector(im_np_scale)
    t1 = time.time()
    fps = int(1 / (t1 - t0))
    if len(faces) > 0:
        for face in faces:
            left = max(face.left(), 0)
            top = max(face.top(), 0)
            right = min(face.right(), im_np_scale.shape[1])
            bottom = min(face.bottom(), im_np_scale.shape[0])

            face_img, bbox = rescale_face(frame, [top, left, bottom, right], scale_size)

            top = bbox[0]
            left = bbox[1]
            bottom = bbox[2]
            right = bbox[3]

            f.write("%d %d %d %d" % (left, top, right, bottom))

    else:
        top = 0
        left = 0
        bottom = 0
        right = 0

        print("Fail to detect face! %s" % image_path)
        f.write("%d %d %d %d" % (left, top, right, bottom))

    f.close()

    # out.release()
    # cv2.destroyAllWindows()


def main():
    # face detection by dlib
    detector = dlib.get_frontal_face_detector()

    folder_types = os.listdir(video_path)
    folder_types.sort()

    for folder_type in folder_types:
        folder_sub_types = os.listdir(os.path.join(video_path, folder_type))
        folder_sub_types.sort()

        for folder_sub_type in folder_sub_types:
            folder_identities = os.listdir(os.path.join(video_path, folder_type, folder_sub_type))
            folder_identities.sort()
            if len(folder_identities) == 0:
                print("%s has no images!!" % os.path.join(video_path, folder_type, folder_sub_type))
                continue
            if os.path.isdir(os.path.join(video_path, folder_type, folder_sub_type, folder_identities[0])):
                for folder_id in folder_identities:
                    print("%s" % os.path.join(video_path, folder_type, folder_sub_type, folder_id))
                    images = glob.glob(os.path.join(video_path, folder_type, folder_sub_type, folder_id, '*.jpg'))
                    images = sorted(images)
                    vid_len = len(images)

                    for i, image_path in enumerate(images):
                        face_dectection(image_path, vid_len, detector)

            else:
                folder_id = folder_sub_type
                print("%s" % os.path.join(video_path, folder_type, folder_id))

                images = glob.glob(os.path.join(video_path, folder_type, folder_id, '*.jpg'))
                images = sorted(images)
                vid_len = len(images)

                for i, image_path in enumerate(images):
                    face_dectection(image_path, vid_len, detector)


if __name__ == '__main__':
    main()