##################################################################################
# # Read different datasets: SiW
# # author: mzh, 21/01/2021, ULR
##################################################################################

import pandas as pd
import cv2
import math
import numpy as np

import matplotlib
matplotlib.use('TKAgg')

import torch
import torch.utils.data as data



class SiW(data.Dataset):
    def __init__(self, video_file, depth_map_size, transform=None):
        self.video_file = video_file
        self.transform = transform
        self.depth_map_size = depth_map_size

        self.df_video = pd.read_csv(self.video_file, sep=',', index_col=None, header=None, names=['image_path', 'bbox', 'spoofing_label', 'type'])

    def __len__(self):
        #print('dataset length %d'%(int(len(self.df_video['image_path']))))
        return int(len(self.df_video['image_path']))

    def __getitem__(self, index):

        image_path = self.df_video.loc[index]['image_path']
        spoofing_label = self.df_video.loc[index]['spoofing_label']
        bbox = self.df_video.loc[index]['bbox']

        strpath = str.split(image_path, 'SiW_release')
        map_path = strpath[0]+'SiW_release/'+'Depth'+strpath[1]
        frame = str.split(image_path, '/')[-1]
        frame = int(frame[:-4])-1
        map_path = map_path[:-8]+'%04d.jpg'%frame

        # print(image_path)
        # print(map_path)
        image_x, map_x = self.get_single_image_x(image_path, map_path, bbox)

        sample = {'image_x':image_x, 'map_x':map_x, 'spoofing_label':spoofing_label, 'image_path':image_path, 'map_path':map_path}
        if self.transform:
            sample = self.transform(sample)
            sample['image_path'] = image_path
            sample['map_path'] = map_path
        return sample

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

    def get_single_image_x(self, image_path, map_path, bbox):

        # random scale from [1.2 to 1.5]
        #face_scale = np.random.randint(12, 15)
        face_scale = np.random.randint(20, 25)
        face_scale = face_scale / 10.0
        #print('face_scale %f'%face_scale)
        # face_scale = 1.0

        image_x = np.zeros((256, 256, 3))
        map_x = np.zeros((self.depth_map_size, self.depth_map_size))
        #map_x = np.zeros((256, 256))

        # RGB
        # image_path = os.path.join(image_path, image_name)
        image_x_temp = cv2.imread(image_path)
        if image_x_temp is None:
            print(image_path)
        # cv2.imwrite('image_x.jpg', image_x_temp)
        #
        # image_x = cv2.resize(self.crop_face_from_scene(image_x_temp, bbox, 1.0), (256, 256))
        # cv2.imwrite('image_x_1.0.jpg', image_x)
        # image_x = cv2.resize(self.crop_face_from_scene(image_x_temp, bbox, 1.5), (256, 256))
        # cv2.imwrite('image_x_1.5.jpg', image_x)
        # image_x = cv2.resize(self.crop_face_from_scene(image_x_temp, bbox, 2.0), (256, 256))
        # cv2.imwrite('image_x_2.0.jpg', image_x)
        # image_x = cv2.resize(self.crop_face_from_scene(image_x_temp, bbox, 2.5), (256, 256))
        # cv2.imwrite('image_x_2.5.jpg', image_x)
        image_x = cv2.resize(self.crop_face_from_scene(image_x_temp, bbox, face_scale), (256, 256))
        #cv2.imwrite('image_x.jpg', image_x)

        ## Holistic face, i.e., enlarge the face region to include more region as in :
        ## []] Määttä, J.; Hadid, A.; Pietikäinen, M. Face spoofing detection from single images using micro-texture
        ##1587 analysis. Biometrics (IJCB), 2011 international joint conference on. IEEE, 2011, pp. 17.
        # ] Patel, K.; Han, H.; Jain, A.K.Secure face unlock: Spoof detection on smartphones.IEEE transactions
        # on 1698 information forensics and security  2016, 11, 22682283.
        ####

        # image_x = cv2.resize(self.crop_face_from_scene(image_x_temp, bbox, face_scale), (256, 256))
        # cv2.imwrite('image_x_original.jpg', image_x)
        #
        # strbbox=str.split(bbox, ' ')
        # xtopleft = int(strbbox[0])
        # ytopleft = int(strbbox[1])
        # xbottomright = int(strbbox[2])
        # ybottomright = int(strbbox[3])
        # scale = 1.5
        #
        # xradii = (xbottomright - xtopleft) / 2
        # yradii = (ybottomright - ytopleft) / 2
        #
        # xcenter = (xbottomright + xtopleft) / 2
        # ycenter = (ybottomright + ytopleft) / 2
        #
        # ysize, xsize,  c = image_x_temp.shape
        # xxtopleft = max(0, xcenter - scale * xradii)
        # yytopleft = max(0, ycenter - scale * yradii)
        # xxbottomright = min(xsize, xcenter + scale * xradii)
        # yybottomright = min(ysize, ycenter + scale * yradii)
        #
        # bbox = ('%d %d %d %d'%(int(xxtopleft), int(yytopleft), int(xxbottomright), int(yybottomright)))
        #
        #
        # image_x = cv2.resize(self.crop_face_from_scene(image_x_temp, bbox, face_scale), (256, 256))

        image_hsv = cv2.cvtColor(image_x, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(image_hsv)
        # s.fill(255)
        # v.fill(255)
        # hsv_image = cv2.merge([h, s, v])
        # out = cv2.cvtColor(hsv_image, cv2.COLOR_HSV2BGR)
        # cv2.imwrite('image_h.jpg', out)
        # h, s, v = cv2.split(image_hsv)
        # h.fill(255)
        # v.fill(255)
        # hsv_image = cv2.merge([h, s, v])
        # out = cv2.cvtColor(hsv_image, cv2.COLOR_HSV2BGR)
        # cv2.imwrite('image_s.jpg', out)
        # h, s, v = cv2.split(image_hsv)
        # h.fill(255)
        # s.fill(255)
        # hsv_image = cv2.merge([h, s, v])
        # out = cv2.cvtColor(hsv_image, cv2.COLOR_HSV2BGR)
        # cv2.imwrite('image_v.jpg', out)
        #
        # cv2.imwrite('image_x_temp.jpg', image_x_temp)
        # cv2.imwrite('image_x.jpg', image_x)

        b, g, r = cv2.split(image_x)
        image_x = cv2.merge([b,g,r,h,s,v])

        # image_YCbCr = cv2.cvtColor(image_x, cv2.COLOR_BGR2YCrCb)
        # Y, Cb, Cr = cv2.split(image_YCbCr)
        # Cb.fill(255)
        # Cr.fill(255)
        # YCbCr_image = cv2.merge([Y, Cb, Cr])
        # out = cv2.cvtColor(YCbCr_image, cv2.COLOR_BGR2YCrCb)
        # cv2.imwrite('image_Y.jpg', out)
        # Y, Cb, Cr = cv2.split(image_YCbCr)
        # Y.fill(255)
        # Cr.fill(255)
        # YCbCr_image = cv2.merge([Y, Cb, Cr])
        # out = cv2.cvtColor(YCbCr_image, cv2.COLOR_BGR2YCrCb)
        # cv2.imwrite('image_Cb.jpg', out)
        # Y, Cb, Cr = cv2.split(image_YCbCr)
        # Y.fill(255)
        # Cb.fill(255)
        # YCbCr_image = cv2.merge([Y, Cb, Cr])
        # out = cv2.cvtColor(YCbCr_image, cv2.COLOR_BGR2YCrCb)
        # cv2.imwrite('image_Cr.jpg', out)
        #
        # cv2.imwrite('image_x_temp.jpg', image_x_temp)
        # cv2.imwrite('image_x.jpg', image_x)
        #
        # image_YCbCr = cv2.cvtColor(image_x, cv2.COLOR_BGR2YCrCb)
        # Y, Cb, Cr = cv2.split(image_YCbCr)
        # Cb.fill(255)
        # Cr.fill(255)
        # YCbCr_image = cv2.merge([Y, Cb, Cr])
        # out = cv2.cvtColor(YCbCr_image, cv2.COLOR_BGR2YCrCb)
        # out = cv2.cvtColor(out, cv2.COLOR_BGR2GRAY)
        # cv2.imwrite('image_gray_Y.jpg', out)
        # Y, Cb, Cr = cv2.split(image_YCbCr)
        # Y.fill(255)
        # Cr.fill(255)
        # YCbCr_image = cv2.merge([Y, Cb, Cr])
        # out = cv2.cvtColor(YCbCr_image, cv2.COLOR_BGR2YCrCb)
        # out = cv2.cvtColor(out, cv2.COLOR_BGR2GRAY)
        # cv2.imwrite('image_gray_Cb.jpg', out)
        # Y, Cb, Cr = cv2.split(image_YCbCr)
        # Y.fill(255)
        # Cb.fill(255)
        # YCbCr_image = cv2.merge([Y, Cb, Cr])
        # out = cv2.cvtColor(YCbCr_image, cv2.COLOR_BGR2YCrCb)
        # out = cv2.cvtColor(out, cv2.COLOR_BGR2GRAY)
        # cv2.imwrite('image_gray_Cr.jpg', out)
        #
        # cv2.imwrite('image_x_temp.jpg', image_x_temp)
        # cv2.imwrite('image_x.jpg', image_x)

        # data augment from 'imgaug' --> Add (value=(-40,40), per_channel=True), GammaContrast (gamma=(0.5,1.5))
        #image_x_aug = seq.augment_image(image_x)
        # image_x_tensor = transform1(image_x) ## toTensor [c, h, w] instead of [h, w, c] in cv2.imread()

        # gray-map
        # map_path = os.path.join(map_path, map_name)
        strmap_path = str.split(map_path, 'Depth')
        map_type = str.split(strmap_path[-1], '/')[2]
        if map_type == 'live':
            map_x_temp = cv2.imread(map_path, 0)

            ## original face region
            map_x = cv2.resize(self.crop_face_from_scene(map_x_temp, bbox, 1.2), (self.depth_map_size, self.depth_map_size))
            #map_x = cv2.resize(self.crop_face_from_scene(map_x_temp, bbox, 1.2), (256, 256))
            #cv2.imwrite('map_x.jpg', map_x)

            # map_x_tensor = transform1(map_x) ## toTensor [c, h, w] instead of [h, w, c] in cv2.imread()

            # map_x_tensor = map_x_tensor.squeeze(0)
        else:
            map_x_tensor = torch.zeros((self.depth_map_size,self.depth_map_size), dtype = torch.float)

        #return image_x_tensor, map_x_tensor
        return image_x, map_x