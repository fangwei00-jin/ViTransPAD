import os
import cv2
import io
import glob
import scipy
import json
import zipfile
import random
import collections
import torch
import math
import numpy as np
import torchvision.transforms.functional as F
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from PIL import Image, ImageFilter
from skimage.color import rgb2gray, gray2rgb
import pandas as pd
import json

from core.utils import ZipReader, create_random_shape_with_random_motion
from core.utils import Stack, ToTorchFormatTensor, GroupRandomHorizontalFlip

debug = 0

class Dataset(torch.utils.data.Dataset):
    def __init__(self, args: dict, dataset, face_scale, split='train', debug=False):
        if split == 'train':
            self.video_file = os.path.join(args['data_root'], args['name'])
        else:
            data_path = os.path.join(args['data_root'], args['name'])
            data_path = data_path.replace("train","test")
            data_path = data_path.replace("Train","Test")
            self.video_file = data_path
            self.video_file = os.path.join(args['data_root'], args['name'])
        self.face_scale = face_scale
        self.dataset = dataset
        self.args = args
        self.split = split
        self.sample_length = args['sample_length']
        self.size = self.w, self.h = (args['w'], args['h'])
        # self.df_video = pd.read_csv(self.video_file, sep=',', index_col=None, header=None,
        #                             names=['image_path', 'bbox', 'spoofing_label', 'type'])
        self.df_video = json.load(open(self.video_file))
        self.video_names = list(self.df_video.keys())
        # # with open(os.path.join(args['data_root'], args['name'], split+'.json'), 'r') as f:
        # with open(os.path.join(args['data_root'], args['name'], split+'.json'), 'r') as f:
        #     self.video_dict = json.load(f)
        # self.video_names = list(self.video_dict.keys())
        # if debug or split != 'train':
        #     self.video_names = self.video_names[:100]

        self._to_tensors = transforms.Compose([
            Stack(),
            ToTorchFormatTensor(), ])

    def __len__(self):
        #return len(self.video_names)
        return int(len(self.df_video))

    def __getitem__(self, index):
        try:
            item = self.load_item(index)
        except:
            print('Loading error in video {}\n'.format(self.video_names[index]))
            item = self.load_item(0)
        return item

    def load_item(self, index):
        item = {}
        if debug:
            # video = "129-2-2-2-1" ##debug
            video = "real_client103_session01_webcam_authenticate_controlled_1_2" ##debug
        else:
            video = self.video_names[index]
        all_frames = self.df_video[video]['frames']
        all_masks = []
        for _ in range(len(all_frames)):
            m = Image.fromarray(np.zeros((self.h, self.w)).astype(np.uint8))
            all_masks.append(m.convert('L'))

        spoofing_label = self.df_video[video]['spoofing_label']
        bboxes = list(self.df_video[video]['bboxes'])

        if debug:
            ref_index = get_ref_index(len(all_frames), len(all_frames)) ##debug
        else:
            ref_index = get_ref_index(len(all_frames), self.sample_length)

        # read video frames
        frames = []
        masks = []
        maps = []

        ## check bbox of the selected frames and re-select the ref_index if necessary ##
        for idx in ref_index:
            # if debug:
            #     idx = 33 ## debug
            image_path = all_frames[idx]
            bbox = bboxes[idx]
            ## check bbox if all coordinates are 0 as [0 0 0 0]
            bbox_num = [int(p) for p in str.split(bbox, " ") if (p.startswith('-') and p[1:] or p).isdigit()]
            if not any(np.array(bbox_num)): ## if there is no a non zero element
                print("bbox is empty in %s!"%image_path)
                raise ValueError("bbox is empty in %s!"%image_path)

            # check bbox if having 4 coordinates of top-left, bottom-right
            xx = str.split(bbox, " ")
            if len([p for p in xx if (p.startswith('-') and p[1:] or p).isdigit()]) < 4:
                print("bbox is not correct: %d %s\n" % (idx, image_path))
                ref_index = []
                ##select new frames ###
                for j in range(len(all_frames)):
                    bbox = bboxes[j]
                    xx = str.split(bbox, " ")
                    if len([p for p in xx if (p.startswith('-') and p[1:] or p).isdigit()]) < 4:
                        continue
                    else:
                        ref_index.append(j)
                        if len(ref_index) == self.sample_length:
                            break
                if len(ref_index) < self.sample_length:
                    raise ValueError("No enough frames can be selected in the video %s"%video)

                break

        for idx in ref_index:
            # if debug:
            #     idx = 33 ## debug
            #     #pass
            image_path = all_frames[idx]
            bbox = bboxes[idx]

            if 'SiW' in self.dataset:
                strpath = str.split(image_path, 'SiW_release')
                map_path = strpath[0]+'SiW_release/'+'Depth'+strpath[1]
                frame = str.split(image_path, '/')[-1]
                frame = int(frame[:-4]) - 1
                map_path = map_path[:-8] + '%04d.jpg' % frame
            ### using resample OuluNPU
            if 'OuluNPU' in self.dataset:
                strpath = str.split(image_path, 'OuluNPU')
                pathstr = str.split(strpath[1], '/')
                video_folder = pathstr[2]
                #video_folder = video_folder[:-2]  ##resample video
                video_folder = video_folder[:8]  ##resample video or original video
                image_path = os.path.join(strpath[0] + 'OuluNPU/', pathstr[1], video_folder, pathstr[3])
                map_path = os.path.join(strpath[0] + 'OuluNPU/' + 'Depth', pathstr[1], video_folder, pathstr[3])
            ### using resample CASIA
            if 'CASIA' in self.dataset and 'resample' in self.dataset:
                strpath = str.split(image_path, 'CASIA')
                pathstr = str.split(strpath[1], '/')
                video_folder = pathstr[3]
                video_folder = video_folder[:-2]  ##resample video
                image_path = os.path.join(strpath[0] + 'CASIA/', pathstr[1], pathstr[2], video_folder, pathstr[4])
                map_path = os.path.join(strpath[0] + 'CASIA/' + 'Depth', pathstr[1], pathstr[2], video_folder, pathstr[4])
            ## using original CASIA
            if 'CASIA' in self.dataset and 'resample' not in self.dataset:                
                strpath = str.split(image_path, 'CASIA')
                map_path = strpath[0]+'CASIA/'+'Depth'+strpath[1]
            ## using resample REPLAY
            if 'REPLAY' in self.dataset and 'resample' in self.args['name']:
                strpath = str.split(image_path, 'REPLAY')
                pathstr = str.split(strpath[1], '/')
                video_folder = pathstr[-2]
                video_folder = video_folder[:-2]  ##resample video
                if pathstr[2] == 'attack':
                    image_path = os.path.join(strpath[0] + 'REPLAY/', pathstr[1], pathstr[2], pathstr[3], video_folder, pathstr[5])
                    map_path = os.path.join(strpath[0] + 'REPLAY/' + 'Depth', pathstr[1], pathstr[2], pathstr[3], video_folder, pathstr[5])
                else:
                    image_path = os.path.join(strpath[0] + 'REPLAY/', pathstr[1], pathstr[2],  video_folder, pathstr[4])
                    map_path = os.path.join(strpath[0] + 'REPLAY/' + 'Depth', pathstr[1], pathstr[2], video_folder, pathstr[4])           
            ### using original REPLAY
            if 'REPLAY' in self.dataset and 'resample' not in self.args['name']:
                strpath = str.split(image_path, 'REPLAY')
                map_path = strpath[0]+'REPLAY/'+'Depth'+strpath[1]


            # if self.dataset == 'OuluNPU':
            #     strpath = str.split(image_path, 'OuluNPU')
            #     map_path = strpath[0]+'OuluNPU/'+'Depth'+strpath[1]
            # if self.dataset == 'CASIA':
            #     strpath = str.split(image_path, 'CASIA')
            #     map_path = strpath[0]+'CASIA/'+'Depth'+strpath[1]
            # if self.dataset == 'REPLAY':
            #     strpath = str.split(image_path, 'REPLAY')
            #     map_path = strpath[0]+'REPLAY/'+'Depth'+strpath[1]


            if debug:
                print(image_path)

            img, map = self.get_single_image_x(image_path, map_path, bbox, spoofing_label)

            frames.append(img)
            masks.append(all_masks[idx])
            maps.append(map)
        if self.split == 'train':
            frames, maps = GroupRandomHorizontalFlip()(frames, maps)
        # To tensors
        frame_tensors = self._to_tensors(frames)*2.0 - 1.0
        maps_tensors = self._to_tensors(maps)*2.0 - 1.0
        mask_tensors = self._to_tensors(masks)

        item['frame_tensors'] = frame_tensors
        item['mask_tensors'] = mask_tensors
        item['map_tensors'] = maps_tensors

        frames_cv = []
        masks_cv = []
        maps_cv = []
        for i in range(len(frames)):
            frames_cv.append(np.array(frames[i])[:, :, ::-1].copy())
            masks_cv.append(np.array(masks[i]).copy())
            maps_cv.append(np.array(maps[i].copy()))

        item['frames'] = torch.tensor(frames_cv)
        item['masks'] = torch.tensor(masks_cv)
        item['maps'] = torch.tensor(maps_cv)
        item['spoofing_labels'] = torch.tensor(spoofing_label)
        return item

    def crop_face_from_scene(self, image,bbox, scale):
        strbbox=str.split(bbox, ' ')
        y1=int(strbbox[1])
        x1=int(strbbox[0])
        y2=int(strbbox[3])
        x2=int(strbbox[2])

        w=x2-x1
        h=y2-y1

        y_mid=(y1+y2)/2.0
        x_mid=(x1+x2)/2.0
        h_img, w_img = image.shape[0], image.shape[1]
        #w_img,h_img=image.size
        w_scale=scale*w
        h_scale=scale*h
        y1=y_mid-h_scale/2.0
        x1=x_mid-w_scale/2.0
        y2=y_mid+h_scale/2.0
        x2=x_mid+w_scale/2.0
        y1=max(math.floor(y1),0)
        x1=max(math.floor(x1),0)
        y2=min(math.floor(y2),h_img)
        x2=min(math.floor(x2),w_img)

        region=image[y1:y2,x1:x2]
        #region=image[x1:x2,y1:y2]
        return region

    def get_single_image_x(self, image_path, map_path, bbox, spoof_label):
        # face_scale = np.random.randint(20, 25)
        if self.face_scale[1] > self.face_scale[0]:
            face_scale = np.random.randint(int(self.face_scale[0] * 10), int(self.face_scale[1] * 10))
            face_scale = face_scale / 10.0
        elif self.face_scale[1] == self.face_scale[0]:
            face_scale = self.face_scale[0]
        else:
            raise ValueError('face_scale[1] should not smaller than face_sacle[0]!')


        if 'SiW' in self.dataset:
            map_x = np.zeros((self.h, self.w))+128 ## SiW: adding 128 used to avoid the tensor of the grounth of the spoof depth map
                                                   ## is same as the tensor of the generated spoof depth map during the early stage.
                                                   ## This results the training of Discriminator can be converged as well as the
                                                   ## training of the Generator.
        if 'OuluNPU' in self.dataset:
            # map_x = np.zeros((self.h, self.w)) ## OuluNPU: The tensor of the ground truth of spoof depth map with the bias of 128
            #                                    ## is too close to the tensor of the predicted spoof depth map, the l1 distance is
            #                                    ## smaller than 1e-2. Using the original ground truth, the distance can be more than 1.0
            map_x = np.zeros((self.h, self.w)) + 128

        if 'CASIA' in self.dataset:
            map_x = np.zeros((self.h, self.w)) + 128

        if 'REPLAY' in self.dataset:
            map_x = np.zeros((self.h, self.w)) + 128

        image_x_temp = cv2.imread(image_path)
        if image_x_temp is None:
            print("Image is None in get_single_image_x() : %s"%image_path)
            raise ValueError("Image is None in get_single_image_x()!")
        image_crop = self.crop_face_from_scene(image_x_temp, bbox, face_scale)
        if len(image_crop)==0:
            print("Crop image is empty in get_single_image_x() : %s"%image_path)
            raise ValueError("Crop image is empty!")

        image_x = cv2.resize(image_crop, (self.w, self.h))

        # gray-map
        # if self.dataset == 'SiW':
        #     strmap_path = str.split(map_path, 'Depth')
        #     map_type = str.split(strmap_path[-1], '/')[2]
        # if self.dataset == 'OuluNPU':
        #     strmap_path = str.split(map_path, 'Depth')
        #     map_type = str.split(strmap_path[-1], '/')[2]
        #     map_type = str.split(map_type, '_')[-1]
        #     map_type = 'live' if map_type=='1' else 'spoof'


        #if map_type == 'live':
        if spoof_label == 1: ## live
            map_x_temp = cv2.imread(map_path, 0)
            if map_x_temp is None:
                print("Depth map is empty in get_single_image_x() : %s" % map_path)
                raise ValueError("Depth map is empty in get_single_image_x()!")
            map_x_temp_crop = self.crop_face_from_scene(map_x_temp, bbox, 1.2)
            if len(map_x_temp_crop) == 0:
                print("Crop depth map is empty in get_single_image_x() : %s" % image_path)
                raise ValueError("Crop depth map is empty!")
            map_x = cv2.resize(map_x_temp_crop, (self.w, self.h))


        img = cv2.cvtColor(image_x, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(img)

        # map = cv2.cvtColor(map_x, cv2.COLOR_BGR2RGB)
        map = Image.fromarray(map_x)
        depth_map = map.convert("RGB")
        return image, depth_map

def get_ref_index(length, sample_length):
    # if random.uniform(0, 1) > 0.5:
    #     ref_index = random.sample(range(length), sample_length)
    #     ref_index.sort()
    # else:
    #     pivot = random.randint(0, length-sample_length)
    #     ref_index = [pivot+i for i in range(sample_length)]
    if sample_length > length:
        raise ValueError("Sampling frames are more than video lenght!")
    ninterval = math.ceil(length/sample_length)
    indicies = list(range(length))
    ref_index = indicies[0:length:ninterval]
    #print(ref_index)
    return ref_index


