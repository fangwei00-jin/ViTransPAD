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
from PIL import Image
from skimage.color import rgb2gray, gray2rgb
import pandas as pd
import json
from pdb import set_trace as bp 

from core.utils import ZipReader, create_random_shape_with_random_motion
from core.utils import Stack, ToTorchFormatTensor, GroupRandomHorizontalFlip
import imgaug.augmenters as iaa

### data augment from 'imgaug' --> Add (value=(-40,40), per_channel=True), GammaContrast (gamma=(0.5,1.5))
### The random seed of iaa is set inside of iaa, which is not affected by the maunual seed set in this script;
seq = iaa.Sequential([
    iaa.AverageBlur(k=(1, 15)), ##Blur each image using a mean over neihbourhoods that have a random size between 2x2 and 11x11
    #iaa.AveragePooling(8), ## Averag pooling as similar as blur
    iaa.Fliplr(0.5), ## left-right flip
    #iaa.JpegCompression(compression=(70, 99)), ## JpegCompression,Remove high frequency components in images via JPEG
    iaa.Add(value=(-40,40), per_channel=True), # Add color
    iaa.GammaContrast(gamma=(0.5,1.5)), # GammaContrast with a gamma of 0.5 to 1.5
    iaa.AddToHueAndSaturation((-50, 50), per_channel=True) ## Increases or decreases hue and saturation by random values.

])


debug = 0

class Dataset(torch.utils.data.Dataset):
    def __init__(self, args: dict, dataset, split='train', debug=False):
        self.video_files = []
        self.video_files_val = []
        self.modalities = []
        self.split = split
        if self.split == 'train':
            #modalities_files = args['train']['modality']  ## modality can be 1 (singel modality) or 2 multimodality
            modalities_files = dataset
            for modality_file in modalities_files:
                modality = str.split(modality_file, '_')[-1][:-5]
                self.modalities.append(modality)

                for modality_file in modalities_files:
                    self.video_files.append(os.path.join(args['train']['data_root'], modality_file))
                self.face_scale = args['train']['face_scale']
        else:
            self.video_files.append(os.path.join(args['test']['test_data']))
            self.video_files_val.append(os.path.join(args['test']['val_data']))
            self.face_scale = args['test']['face_scale']

        self.args = args
        self.classes = self.args['data_loader']['class']
        self.sample_length = args['data_loader']['sample_length']
        self.size = self.w, self.h = (args['data_loader']['w'], args['data_loader']['h'])
        # self.df_video = pd.read_csv(self.video_files, sep=',', index_col=None, header=None,
        #                             names=['image_path', 'bbox', 'spoofing_label', 'type'])
        self.df_video = json.load(open(self.video_files[0])) ## first modality video file
        if len(self.modalities)>1:
            self.df_video1 = json.load(open(self.video_files[1])) ## second modality video file
        self.video_names = list(self.df_video.keys())
        #random.shuffle(self.val_video_names)
        # # with open(os.path.join(args['data_root'], args['name'], split+'.json'), 'r') as f:
        # with open(os.path.join(args['data_root'], args['name'], split+'.json'), 'r') as f:
        #     self.video_dict = json.load(f)
        # self.video_names = list(self.video_dict.keys())
        # if debug or split != 'train':
        #     self.video_names = self.video_names[:100]

        attack_vid_cnt = 0
        bonafide_vid_cnt = 0
        for vid in self.video_names:
            if str.split(vid, '_')[-2] != '0':
                attack_vid_cnt += 1
            if str.split(vid, '_')[-2] == '0':
                bonafide_vid_cnt += 1


        self._to_tensors = transforms.Compose([
            Stack(),
            ToTorchFormatTensor(), ])

    def __len__(self):

        if self.split=='train':
            return int(len(self.df_video)) ## total videos for training = batch * batch_size
            #return  1## debug
        else:
            #return int(len(self.df_video)) ## total videos for test = batch * batch_size
            return 10  ## debug



    def __getitem__(self, index):
        try:
            item = self.load_item(index)
            # ## debug
            # print(item["video"])
        except:
            print('Loading error in video {}\n'.format(self.video_names[index]))
            item = self.load_item(0)
            # ## debug
            # print(item["video"])
        return item

    def load_item(self, index):
        item = {}
        if debug:
            video = "129-2-2-2-1" ##debug
            ##video = "real_client103_session01_webcam_authenticate_controlled_1_2" ##debug
        else:
            video = self.video_names[index]
        all_frames = self.df_video[video]['frames'] ##RGB images
        all_masks = []
        for _ in range(len(all_frames)):
            ## Mask==0 is the ROI zone;
            ## in this work, all zone is the ROI zone, thus the enitre mask as large as image is 0.
            m = Image.fromarray(np.zeros((self.h, self.w)).astype(np.uint8))
            all_masks.append(m.convert('L'))
        if self.classes == 2:
            spoofing_label = self.df_video[video]['spoofing_label'] ## 1 Bonafide | 0 Attack
        else:
            if "WMCA" in self.args['train']['data_root'] or "WMCA" in self.args['test']['test_data']:
                spoofing_label = int(str.split(video, '_')[-2]) ## 0 Bonafide, 1 Glasses, 2 Fake face (mask), 3 Printed, 4 Video
                type_id = int(str.split(video, '_')[-1])
                if spoofing_label == 3 and type_id in [5,10,15]:
                    spoofing_label = 4 # Electornic printed is considered as a video
            else:
                raise ValueError("Only WMCA support the detection of attack type!")
        bboxes = list(self.df_video[video]['bboxes'])

        if debug:
            ref_index = self.get_ref_index(len(all_frames), len(all_frames)) ##debug
        else:
            if len(all_frames) < self.sample_length:
                print('%s has not enought sampling frames!'%video)
            ref_index = self.get_ref_index(len(all_frames), self.sample_length)
        # read video frames
        frames = []
        masks = []
        maps = []
        frames_path = []

       
        for idx in ref_index:
            # if debug:
            #     idx = 33 ## debug
            
            image_path = all_frames[idx]
            image_path = self.update_image_path(image_path)


            bbox = bboxes[idx]
            ## check bbox if all coordinates are 0 as [0 0 0 0]
            bbox_num = [int(p) for p in str.split(bbox, " ") if (p.startswith('-') and p[1:] or p).isdigit()]
            if not any(np.array(bbox_num)): ## if there is no a non zero element
                print("bbox is empty in %s!"%image_path)
                raise ValueError("bbox is empty in %s!"%image_path)

        for idx in ref_index:

            bbox = bboxes[idx]

            ##first modality
            image_path = all_frames[idx]
            image_path = self.update_image_path(image_path)
            ##print("%s"%image_path) # debug

            img = self.get_single_image(image_path, bbox, spoofing_label)
            ##second modality
            if len(self.modalities) > 1:
                map_path = self.df_video1[video]['frames'][idx]
                map_path = self.update_image_path(map_path)

                map = self.get_single_image(map_path, bbox, spoofing_label)
                maps.append(map)

            if debug:
                print(image_path)

            frames.append(img)
            masks.append(all_masks[idx])
            frames_path.append(image_path)

        if self.split == 'train':
            if len(self.modalities) > 1:
                ListFlip = GroupRandomHorizontalFlip()([frames, maps])
                frames = ListFlip[0]
                maps = ListFlip[1]

        # To tensors
        frame_tensors = self._to_tensors(frames)*2.0 - 1.0
        mask_tensors = self._to_tensors(masks)

        item['frame_tensors'] = frame_tensors
        item['mask_tensors'] = mask_tensors


        # frames_cv = []
        # masks_cv = []

        # for i in range(len(frames)):
        #     frames_cv.append(np.array(frames[i])[:, :, ::-1].copy())
        #     masks_cv.append(np.array(masks[i]).copy())


        # item['frames'] = torch.tensor(frames_cv)
        # item['masks'] = torch.tensor(masks_cv)
        item['spoofing_labels'] = torch.tensor(spoofing_label)
        item['images_path'] = frames_path
        frames_array = []
        for frame in frames:
            frames_array.append(np.array(frame))
        item['frames'] = frames_array

        if len(self.modalities) > 1:
            maps_tensors = self._to_tensors(maps) * 2.0 - 1.0
            item['map_tensors'] = maps_tensors
            # maps_cv = []
            # for i in range(len(frames)):
            #     maps_cv.append(np.array(maps[i].copy()))
            # item['maps'] = torch.tensor(maps_cv)

        ##reseve for debug
        # item['video'] = video
        # item['frames_files'] = all_frames
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

    def get_single_image(self, image_path, bbox, spoof_label):
        if self.split == "train":
            # face_scale = np.random.randint(20, 25)
            if self.face_scale[1] > self.face_scale[0]:
                face_scale = np.random.randint(int(self.face_scale[0] * 10), int(self.face_scale[1] * 10))
                face_scale = face_scale / 10.0
            elif self.face_scale[1] == self.face_scale[0]:
                face_scale = self.face_scale[0]
            else:
                raise ValueError('face_scale[1] should not smaller than face_sacle[0]!')

        else:
            face_scale = self.face_scale


        if '3ddfa' in image_path: ##3D depth map
            if spoof_label == 0: ## spoof
                image_x = np.zeros((self.h, self.w)) + 128
            else: ## live
                image_x = cv2.imread(image_path)
        else:
            image_x = cv2.imread(image_path)

        if image_x is None:
            print("Image is empty in get_single_image_x() : %s" % image_path)
            raise ValueError("Image is empty in get_single_image_x()!")

        ### data augmentation ###
        if self.split == "train":
            image_x_aug=seq.augment_image(image_x)
        else:
            image_x_aug = image_x

        image_x_crop = self.crop_face_from_scene(image_x_aug, bbox, face_scale)
        if len(image_x_crop) == 0:
            print("Crop depth map is empty in get_single_image_x() : %s" % image_path)
            raise ValueError("Crop depth map is empty!")


        image_x = cv2.resize(image_x_crop, (self.w, self.h))
        if len(image_x.shape)==3:
            image_x = cv2.cvtColor(image_x, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(image_x)
        if len(image_x.shape)==2:
            image = image.convert("RGB")

        return image

    def update_image_path(self, image_path):
                
        if "WMCA" in self.args['train']['data_root'] or "WMCA" in self.args['test']['test_data']:
            key_str = str.split(image_path, "datasets")
            dataset_dir = str.split(self.args['train']['data_root'], 'datasets')
            image_path_updated = dataset_dir[0]+'datasets'+key_str[1]
        elif "OCIM" in self.args['config']:
            key_str=str.split(image_path, 'Anti-spoof')
            dataset_dir=str.split(self.args['train']['data_root'], 'Anti-spoof')
            image_path_updated=dataset_dir[0] + 'Anti-spoof' + key_str[1]
        else:
            if "TRAIN" in image_path.upper():
                dataset_dir = self.args['train']['data_root']
                key_str = str.split(dataset_dir, '/')[-1]
                data_root_dir = dataset_dir[:-len(key_str)]
            else:
                dataset_dir = self.args['test']['test_data']
                key_str = str.split(dataset_dir, '/')[-2]
                data_root_dir = dataset_dir[:dataset_dir.index(key_str)]

            ## updat the image_path accoding to current root dir ###
            image_relative_path = str.split(image_path, key_str)[1]
            image_path_updated = data_root_dir+key_str+image_relative_path

        return image_path_updated


    def get_ref_index(self, length, sample_length):
        # if self.split == 'train':
        #     if random.uniform(0, 1) > 0.5:
        #         ref_index = random.sample(range(length), sample_length)
        #         ref_index.sort()
        #     else:
        #         pivot = random.randint(0, length-sample_length)
        #         ref_index = [pivot+i for i in range(sample_length)]
        # else:
        #     pivot=0
        #     ref_index=list(range(pivot, length))
        #     #select_groups = int(length/sample_length) ## for infer_neighbour()
        #     select_groups=1 ## for infer()
        #     ref_index = ref_index[:sample_length*select_groups]

        # if self.split == 'test':
        #     pivot=0
        #     interval = 1
        #     ref_index=list(range(pivot, length, interval))
        #     #select_groups = int(length/sample_length) ## for infer_neighbour()
        #     select_groups=1 ## for infer()
        #     ref_index = ref_index[:sample_length*select_groups]


        # #### random interval and start for sampling frames of given length
        # pivot = random.randint(0, length - sample_length)
        # max_interval = int((length-pivot)/sample_length)
        # interval = random.randint(0, max_interval)
        # ref_index = [pivot+i*interval for i in range(sample_length)]
        #if "OCIM" in self.args['train']['data_root'] or "OuluNPU" in self.args['train']['data_root']:  ## Random sampling video clips
        # if "OuluNPU" in self.args['train']['data_root']:
        #     interval=3  ## sampling interval 3*1/30=0.1 second for a video with 30fps
        #     if self.split == 'train':
        #         nn = random.randint(0, 10000)/10000
        #         if nn >0.5: ## sampling the frames in the original order
        #             pivot=random.randint(0, length - interval * (sample_length - 1) - 1)  ## training randomly sampling the video clip
        #             ref_index=list(range(pivot, length, interval))
        #             ref_index=ref_index[:sample_length]
        #         else: ## sampling the frames in the random order
        #             ref_index=list(range(0, length, interval))
        #             ref_index=random.sample(ref_index, sample_length)
        #
        #     else:  ## test  all video clips sampled sequentially in test videos with the same sample_lenth as training
        #         pivot=0
        #         ref_index=list(range(pivot, length, interval))
        #         #select_groups = 3 ## OCIM
        #         select_groups = 1 ## OuluNPU
        #         ref_index = ref_index[:sample_length*select_groups]
        #         # for i in range(len(ref_index_all))[::sample_length]:
        #         #     if (i + sample_length) <= length:
        #         #         ref_index.append(ref_index_all[i:i + sample_length])
        #         #     else:
        #         #         return ref_index

        # if "OCIM" in self.args['train']['data_root']:  ## Random sampling video clips
        #     interval=3  ## sampling interval 3*1/30=0.1 second for a video with 30fps
        #     ref_index=[]
        #     if self.split == 'train':
        #         pivot=random.randint(0, length - interval * (sample_length - 1) - 1)  ## training randomly sampling the video clip
        #         ref_index=list(range(pivot, length, interval))
        #         ref_index=ref_index[:sample_length]  ## take the
        #     else:  ## test  all video clips sampled sequentially in test videos with the same sample_lenth as training
        #         pivot=0
        #         ref_index=list(range(pivot, length, interval))
        #         select_groups = 3
        #         ref_index = ref_index[:sample_length*select_groups]
        #         # for i in range(len(ref_index_all))[::sample_length]:
        #         #     if (i + sample_length) <= length:
        #         #         ref_index.append(ref_index_all[i:i + sample_length])
        #         #     else:
        #         #         return ref_index
        # elif "SiW" in self.args['train']['data_root']: ## Random sampling video clips
        # if "WMCA" in self.args['train']['data_root']: ## Random sampling video clips
        #     #### SiW: Randomly sampling (3 frames interval) continuous 8/16/32/40 frames in video of 15s~38s/30fps or 60fps ####
        #     interval = 3 ## sampling interval 3*1/30=0.1 second for a video with 30fps
        #     if self.split == 'train': ## Random sampling video clip with sample_length frames
        #         pivot = random.randint(0, length - interval*(sample_length-1)-1) ## training randomly sampling the video clip
        #     else: ## test only sample_length the first video clip with sample_length frames
        #         pivot = 0 ## fixing the sampling video clip for test  to have the same test data
        #     ref_index = list(range(pivot,length,interval))
        #     ref_index = ref_index[:sample_length] ## take the first sample_length, i.e. sampling sample_length*0.1 second video clip for a 30fps video
        # else:
        #     #### OuluNPU/WMCA: Sampling train_total_samples frames (sampling indices are always same for each training iteration) as the representation of a video ####
        #     train_total_samples = 40 ## total sampling 40 frames for each train video
        #     interval = int(length/train_total_samples) ## total sampling 40 frames for 5 seconds
        #     interval = 1 if interval < 1 else interval
        #
        #     ref_index = list(range(0,length,interval))
        #     ref_index = ref_index[:sample_length] ## take the first sample_length, i.e. first s seconds frames
        #     if len(ref_index) < sample_length:
        #         print(ref_index)
        #         ref_index = ref_index+ref_index*int((sample_length/len(ref_index)))
        #         ref_index = ref_index[:sample_length]

        ########## Random sampling for training #############
        if self.split == 'train':
            if random.uniform(0, 1) > 0.5: ### first sample_length frames with the same interval
                train_total_samples=40  ## total sampling 40 frames for each train video
                interval=int(length / train_total_samples)
                interval=1 if interval < 1 else interval

                ref_index=list(range(0, length, interval))
                ref_index=ref_index[:sample_length]  ## take the first sample_length, i.e. first s seconds frames
                if len(ref_index) < sample_length:
                    print(ref_index)
                    ref_index=ref_index + ref_index * int((sample_length / len(ref_index)))
                    ref_index=ref_index[:sample_length]
            else: ### randomly select sample_length frames with random interval
                ref_index = random.sample(range(length), sample_length)
                ref_index.sort()
        else:
            test_total_samples = 40 ## total sampling 40 frames for each train video,  first sample_length frames with the same interval for test
            # interval = 3 ## debug
            # select_groups = 5 ## debug
            interval = int(length/test_total_samples)
            interval = 1 if interval < 1 else interval

            ref_index = list(range(0,length,interval))
            ref_index = ref_index[:sample_length] ## take the first sample_length, i.e. first s seconds frames
            # ref_index = ref_index[:sample_length*select_groups] ## debug
            if len(ref_index) < sample_length:
                print(ref_index)
                ref_index = ref_index+ref_index*int((sample_length/len(ref_index)))
                ref_index = ref_index[:sample_length]

        return ref_index

