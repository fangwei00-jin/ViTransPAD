import json
import os
from PIL import Image, ImageFilter
import numpy as np
from core.dataset import Dataset
import torchvision.transforms as transforms
import torch
import importlib
import cv2
from sklearn.metrics import roc_curve, auc
import math
from pdb import set_trace as bp
import random
import os
import matplotlib
matplotlib.use('Agg')
#matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

class Evaluate():
    def __init__(self, config, split, model=None):
        self.log_dir = config['save_dir']
        self.dataset = config['dataset']
        self.face_scale = config['test']['face_scale']
        if split == 'test':
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            self.net = importlib.import_module('model.' + config['test']['model'])
            self.model = self.net.InpaintGenerator(config['test']['Transformer_layers'],
                                                   config['test']['Transformer_heads'], 
                                                   config['test']['channel']).to(self.device)
        else:
            self.model = model
            # self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            # self.device = config['device']
        self.model.eval()

        ### validation videos ###
        self.val_videos_num = config['test']['val_videos_num']
        # train_data_root = config['data_loader']['data_root'][:-5]+'Train/'
        # train_protocol_name = 'train'+config['data_loader']['name'][4:]
        # self.train_video_file = os.path.join(config['data_loader']['data_root'], config['data_loader']['name'])
        # self.train_df_video = json.load(open(self.train_video_file))
        self.val_video_file = config['test']['val_data']
        self.val_df_video = json.load(open(self.val_video_file))
        self.val_video_names = list(self.val_df_video.keys())
        random.shuffle(self.val_video_names)

        ### test videos ###
        self.video_file = os.path.join(config['test']['test_data'])
        self.df_video = json.load(open(self.video_file))
        self.video_names = list(self.df_video.keys())
        #random.shuffle(self.video_names)
        self.w = config['data_loader']['w']
        self.h = config['data_loader']['h']
        self.neighbor_stride = config['test']['neighbor_stride']
        self.fps = config['test']['fps']
        self.ref_length = config['test']['ref_length']
        self.size = self.w, self.h
        self.test_video_sample_interval = config['test']['test_video_sample_interval']
        self.test_frames_sample_interval = config['test']['test_frames_sample_interval']
        self._to_tensors = transforms.Compose([
            Stack(),
            ToTorchFormatTensor(), ])

    def read_frame_from_video(self, df_video, vidname):
        all_frames = df_video[vidname]['frames']
        bboxes = list(df_video[vidname]['bboxes'])
        spoofing_label = df_video[vidname]['spoofing_label']
        video_length = len(all_frames)
        frames = []
        maps = []
        images_path = []
        for idx in range(video_length)[::self.test_frames_sample_interval]:

            image_path = all_frames[idx]
            bbox = bboxes[idx]
            # print(image_path)
            xx = str.split(bbox, " ")
            if len([p for p in xx if (p.startswith('-') and p[1:] or p).isdigit()]) <4:
            # if len([p for p in xx if p != ' ']) <4:
                continue
            # strpath = str.split(image_path, 'SiW_release')
            # map_path = strpath[0] + 'SiW_release/' + 'Depth' + strpath[1]
            # frame = str.split(image_path, '/')[-1]
            # frame = int(frame[:-4]) - 1
            # map_path = map_path[:-8] + '%04d.jpg' % frame

            if 'SiW' in image_path:
                strpath = str.split(image_path, 'SiW_release')
                map_path = strpath[0]+'SiW_release/'+'Depth'+strpath[1]
                frame = str.split(image_path, '/')[-1]
                frame = int(frame[:-4]) - 1
                map_path = map_path[:-8] + '%04d.jpg' % frame

            if 'OuluNPU' in image_path:
                strpath = str.split(image_path, 'OuluNPU')
                map_path = strpath[0]+'OuluNPU/'+'Depth'+strpath[1]

            if 'REPLAY' in image_path:
                if 'REPLAY' in str.split(image_path, '/'):
                    strpath = str.split(image_path, 'REPLAY')
                    map_path = strpath[0]+'REPLAY/'+'Depth'+strpath[1]
                else:
                    strpath = str.split(image_path, 'CASIA')
                    map_path = strpath[0]+'CASIA/'+'Depth'+strpath[1]

            if 'CASIA' in image_path:
                if 'CASIA' in str.split(image_path, '/'):
                    strpath = str.split(image_path, 'CASIA')
                    map_path = strpath[0]+'CASIA/'+'Depth'+strpath[1]
                else:
                    strpath = str.split(image_path, 'REPLAY')
                    map_path = strpath[0]+'REPLAY/'+'Depth'+strpath[1]
            # if 'AriadNext' in self.dataset:
            #     strpath = str.split(image_path, 'AriadNext')
            #     map_path = strpath[0]+'AriadNext/'+'Depth'+strpath[1]
            if  'AriadNext' in image_path:
                strpath = str.split(image_path, 'AriadNext')
                strtmp = str.split(strpath[1], 'train')
                map_path = strpath[0]+'AriadNext'+strtmp[0]+'Depth'+strtmp[1]

            if 'MSU-MFSD' in image_path:
                strpath = str.split(image_path, 'MSU-MFSD')
                map_path = strpath[0] + 'MSU-MFSD/' + 'Depth' + strpath[1]

            img, map = self.get_single_image_x(image_path, map_path, bbox, spoofing_label)

            frames.append(img)
            maps.append(map)
            images_path.append(image_path)

        return  frames, maps, images_path

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

    def get_single_image_x(self, image_path, map_path, bbox, spoofing_label):

        # face_scale = np.random.randint(20, 25) # mode training
        # face_scale = face_scale / 10.0
        #face_scale = 2.0 # mode evaluation
        face_scale = self.face_scale

        #map_x = np.zeros((self.h, self.w))+128 ##
        map_x = np.zeros((self.h, self.w)) ## map_x +128 only used for training

        image_x_temp = cv2.imread(image_path)
        if image_x_temp is None:
            print(image_path)
        image_x = cv2.resize(self.crop_face_from_scene(image_x_temp, bbox, face_scale), (self.w, self.h))

        # gray-map
        # #SiW
        # strmap_path = str.split(map_path, 'Depth')
        # map_type = str.split(strmap_path[-1], '/')[2]
        # if map_type == 'live':
        if spoofing_label == 1:  ## live
            map_x_temp = cv2.imread(map_path, 0)
            map_x = cv2.resize(self.crop_face_from_scene(map_x_temp, bbox, 1.2), (self.w, self.h))


        img = cv2.cvtColor(image_x, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(img)

        # map = cv2.cvtColor(map_x, cv2.COLOR_BGR2RGB)
        map = Image.fromarray(map_x)
        depth_map = map.convert("RGB")
        return image, depth_map

    # sample reference frames from the whole video
    def get_ref_index(self, neighbor_ids, length):
        ref_index = []
        # ref_interval = length//self.ref_length
        for i in range(0, length, self.ref_length):
            if not i in neighbor_ids:
                ref_index.append(i)

        return ref_index

    def infer(self, faceframes, masks, device):
        video_length = len(faceframes)
        y_enc_video = [None] * video_length
        feats_tensors = self._to_tensors(faceframes).unsqueeze(0)*2.0 - 1.0

        mask_tensors = self._to_tensors(masks).unsqueeze(0)

        ## calculate the encoder feats one time, but need large gpu memory (being difficult if video lenght >1k images)
        # feats_tensors, mask_tensors = feats_tensors.to(device), mask_tensors.to(device)        #
        # with torch.no_grad():
        #     feats = self.model.encoder((feats_tensors * (1 - mask_tensors).float()).view(video_length, 3, self.h, self.w))
        #     _, c, feat_h, feat_w = feats.size()
        #     feats = feats.view(1, video_length, c, feat_h, feat_w)

        # completing holes by spatial-temporal transformers
        for f in range(0, video_length, self.neighbor_stride):  ## overlap neighbor_stride frames
        #for f in range(0, video_length, 2*self.neighbor_stride):   ## no overlay
        #for f in range(0, video_length, 1):  ## overlap  2$neighbor_stride-2 frames

            neighbor_ids = [i for i in
                            range(max(0, f - self.neighbor_stride), min(video_length, f + self.neighbor_stride + 1))]
            ref_ids = self.get_ref_index(neighbor_ids, video_length)
            # print("_________len(ref_ids): %d"%len(ref_ids))

            with torch.no_grad():
                ## calculate the encoder feats of each video clip which do not require large gpu memoery but with low efficience
                feats_tensors_clip = feats_tensors[0, neighbor_ids + ref_ids, :, :, :]
                mask_tensors_clip = mask_tensors[0, neighbor_ids + ref_ids, :, :, :]
                feats_tensors_clip, mask_tensors_clip = feats_tensors_clip.to(device), mask_tensors_clip.to(device)
                feats = self.model.encoder(
                    (feats_tensors_clip * (1 - mask_tensors_clip).float()).view(len(neighbor_ids+ref_ids), 3, self.h, self.w))

                # ## predict enc features with encoder feats calculated in one time
                # y_enc = self.model.infer(
                #     feats[0, neighbor_ids + ref_ids, :, :, :], mask_tensors[0, neighbor_ids + ref_ids, :, :, :])
                ## predict enc features
                y_enc = self.model.infer(feats, mask_tensors_clip)

                # pred_imgs = torch.tanh(self.model.decoder(
                #     pred_feat[:len(neighbor_ids), :, :, :])).detach()
                y_enc = y_enc[:len(neighbor_ids)]
                # pred_imgs = (pred_imgs + 1) / 2
                # pred_imgs = pred_imgs.cpu().permute(0, 2, 3, 1).numpy() * 255
                for i in range(len(neighbor_ids)):
                    idx = neighbor_ids[i]
                    embedding = y_enc[i]
                    if y_enc_video[idx] is None:
                        y_enc_video[idx] = embedding
                    else:
                        y_enc_video[idx] = y_enc_video[idx] * 0.5 + embedding * 0.5

        # return pred_depth_maps
        return y_enc_video

    def cal_prob(self, y):
        ## softmax classification probablity
        y_exp = torch.exp(y)
        y_exp_sum = torch.sum(y_exp, 1)
        prob_cls = y_exp / torch.unsqueeze(y_exp_sum, 1)
        spoof_label = torch.argmax(y, dim=1)
        # bp()
        prob_estimate = prob_cls[0][spoof_label]

        return prob_estimate, spoof_label

    ## Threshold can be selected in terms of different criterion
    ## 1) "Best threshold": threshold[i] should let [fpr[i], tpr[i]] most close to [0,1]
    ##         i = np.argmin[fpr**2+(1-tpr)**2]
    ## 2) "Equal Error Rate": threshold[i] let the False Acceptance Rate (FAR) = False Rejection Rate (FRR)
    def get_err_threhold(self, fpr, tpr, threshold):

        # RightIndex = (tpr + (1 - fpr) - 1);
        # right_index = np.argmax(RightIndex)
        # best_th = threshold[right_index]
        # err = fpr[right_index]

        # differ_tpr_fpr_1 = tpr + fpr - 1.0  ## best threshold make tpr -> 1 and fpr -> 0, thus tpr+fpr-1->0
        differ_tpr_fpr_1 = 1.0-tpr + fpr  ## best threshold make tpr -> 1 and fpr -> 0, thus tpr+fpr-1->0
        # right_index = np.argmin(differ_tpr_fpr_1)  ## this is the upper bound  of the threshold based on the current dataset
        idx =np.argsort(differ_tpr_fpr_1)
        threshold_upperbound, threshold_lowerbound = threshold[idx[0]], threshold[idx[0]+1]
        # print("Threshold[right_index]: %f right index %d" % (threshold[right_index], right_index))

        # right_index = np.argmin(fpr ** 2 + (1 - tpr) ** 2)  ## Best threshold
        # print("Threshold[right_index]: %f right index %d" % (threshold[right_index], right_index))

        # best_th = threshold[right_index]
        # err = fpr[right_index]

        # print(err, best_th)
        # return err, best_th
        return threshold_upperbound, threshold_lowerbound

    def performance (self, scores, labels, threshold=0.5):
        idx_type1 = [i for i, s in enumerate(scores) if s == 0 and labels[i] == 1] ## false negative
        idx_type2 = [i for i, s in enumerate(scores) if s == 1 and labels[i] == 0] ## false positive
        type1 = len(idx_type1) ## False negative (False attack)
        type2 = len(idx_type2)  ## False positive (False live)

        count = len(labels)
        num_real = sum(labels)
        num_fake = count - num_real
        ACC = 1 - (type1 + type2) / count
        if num_fake == 0:
            APCER = 0
        else:
            APCER = type2 / num_fake ## false liveness rate/false acceptance rate

        if num_real == 0:
            BPCER = 0
        else:
            BPCER = type1 / num_real ## false attack rate/ false rejection rate

        ACER = (APCER + BPCER) / 2.0

        return ACC, APCER, BPCER, ACER


    def get_depth_map_score(self, df_video, videos, device, ep, threshold=-1):
        # map_score_list = []
        score_list = []
        spoof_list = []
        pred_list_vid = []
        spoof_list_vid =[]
        ep_dir = os.path.join(self.log_dir, 'ep_%03d' % ep)
        if not os.path.isdir(ep_dir):
            os.mkdir(ep_dir)

        if threshold != -1:

            APCE_dir=os.path.join(self.log_dir, ep_dir, 'APCE')
            if not os.path.isdir(APCE_dir):
                os.mkdir(APCE_dir)
            BPCE_dir=os.path.join(self.log_dir, ep_dir, 'BPCE')
            if not os.path.isdir(BPCE_dir):
                os.mkdir(BPCE_dir)

        for i, video in enumerate(videos):

            ### debug ###
            # if threshold != -1:
            #     video = '1-3-46-1' ## false positive case
            # # video = '1-3-46-1'
            # ### debug ###

            print("%d/%d: %s"%(i,len(videos), video))
            faces, maps, images_path = self.read_frame_from_video(df_video, video)
            spoofing_label = df_video[video]['spoofing_label']

            masks = []
            for _ in range(len(faces)):
                m = Image.fromarray(np.zeros((self.h, self.w)).astype(np.uint8))
                masks.append(m.convert('L'))
            # bp()
            # pred_depth_maps = self.infer(faces, masks, device)
            y_transformer = self.infer(faces, masks, device)
            y_exp = [torch.exp(x) for x in y_transformer]
            y_exp_sum = [torch.sum(x) for x in y_exp]
            prob_cls = [x/y_exp_sum[i] for i, x in enumerate(y_exp)]
            scores_vid = [float(x[spoofing_label].cpu()) for x in prob_cls]
            score_list += scores_vid
            spoof_list += [spoofing_label] * len(faces)

            #video-wise prediction
            pred_frames = [float(np.argmax(x.cpu())) for x in prob_cls]
            pred_vid = 1 if sum(pred_frames) > len(pred_frames)/2.0 else 0
            pred_list_vid.append(pred_vid)
            spoof_list_vid.append(spoofing_label)

            false_live = 0
            false_attack = 0
            for k, score in enumerate(scores_vid):
                image_name = str.split(images_path[k], '/')[-1]
                face_frm = int(str.split(image_name, '.')[0])
                face = faces[k]
                face_cv = np.array(face)
                face_cv = face_cv[:, :, ::-1].copy()
                if score < 0.5 and spoofing_label == 0:
                    cv2.imwrite(os.path.join(APCE_dir, "%s_%04d_face.jpg" % (video, face_frm)), face_cv)
                    false_live += 1
                elif score < 0.5 and spoofing_label == 1:
                    cv2.imwrite(os.path.join(BPCE_dir, "%s_%04d_face.jpg" % (video, face_frm)), face_cv)
                    false_attack += 1
                else:
                    cv2.imwrite(os.path.join(ep_dir, "%s_%04d_face.jpg" % (video, face_frm)), face_cv)

            if false_live > 0:
                print("False live images: %d"%false_live)

            if false_attack > 0:
                print("False attack images: %d"%false_attack)


        #return score_list, spoof_list
        return pred_list_vid, spoof_list_vid

    def evaluate(self, ep, ckpt):
        data = torch.load(ckpt, map_location=self.device)
        self.model.load_state_dict(data['netG'])
        print('loading from: {}'.format(ckpt))
        videos = self.video_names[::self.test_video_sample_interval]
        videos_length = len(videos)

        # videos_val_length = self.val_videos_num#200 ## validation videos number
        # val_videos = self.val_video_names[:videos_val_length]#self.train_video_names[:videos_val_length]


        # ### val for calculating threshold ###
        # map_score_list, spoof_list = self.get_depth_map_score(self.val_df_video, val_videos, self.device, ep)
        # #bp()
        # fprs, tprs, thresholds = roc_curve(spoof_list, map_score_list, pos_label=1, drop_intermediate=False)
        # threshold_upperbound, threshold_lowerbound = self.get_err_threhold(fprs, tprs, thresholds)
        # #threshold = (threshold_upperbound + threshold_lowerbound)/2
        # threshold = (threshold_upperbound)/2 ## threshold_lowerbound will be very close to threshold_upperbound
        #                                      ## is if there are false liveness (fp) results
        # print("Threshold is %.04f, threshold_upperbound is %.04f, threshold_lowerbound is %.04f,"%(threshold, threshold_upperbound, threshold_lowerbound))

        ### test ###
        score_list, spoof_list = self.get_depth_map_score(self.df_video, videos, self.device, ep, 0.5)

        ACC, APCER, BPCER, ACER = self.performance(score_list, spoof_list)

        # ## threshold robustness test ##
        # thresholds_interp0 = np.linspace(threshold_lowerbound, threshold_upperbound, num=21, endpoint=True)
        # thresholds_interp1 = np.linspace(thresholds[0]-1+0.1, thresholds[1], num=6, endpoint=True)
        # acc_list = []
        # thresholds_test = list(thresholds_interp0) + list(thresholds_interp1) + list(thresholds[1:])
        # thresholds_test = list(np.sort(thresholds_test))
        # for threshold_test in thresholds_test:
        #     acc, _, _, _= self.performance(map_score_list, spoof_list, threshold_test)
        #     acc_list.append(acc)

        #return ACC, APCER, BPCER, ACER, threshold, fprs, tprs, thresholds, acc_list, thresholds_test
        return ACC, APCER, BPCER, ACER

class Stack(object):
    def __init__(self, roll=False):
        self.roll = roll

    def __call__(self, img_group):
        mode = img_group[0].mode
        if mode == '1':
            img_group = [img.convert('L') for img in img_group]
            mode = 'L'
        if mode == 'L':
            return np.stack([np.expand_dims(x, 2) for x in img_group], axis=2)
        elif mode == 'RGB':
            if self.roll:
                return np.stack([np.array(x)[:, :, ::-1] for x in img_group], axis=2)
            else:
                return np.stack(img_group, axis=2)
        else:
            raise NotImplementedError(f"Image mode {mode}")

class ToTorchFormatTensor(object):
    """ Converts a PIL.Image (RGB) or numpy.ndarray (H x W x C) in the range [0, 255]
    to a torch.FloatTensor of shape (C x H x W) in the range [0.0, 1.0] """

    def __init__(self, div=True):
        self.div = div

    def __call__(self, pic):
        if isinstance(pic, np.ndarray):
            # numpy img: [L, C, H, W]
            img = torch.from_numpy(pic).permute(2, 3, 0, 1).contiguous()
        else:
            # handle PIL Image
            img = torch.ByteTensor(
                torch.ByteStorage.from_buffer(pic.tobytes()))
            img = img.view(pic.size[1], pic.size[0], len(pic.mode))
            # put it from HWC to CHW format
            # yikes, this transpose takes 80% of the loading time/CPU
            img = img.transpose(0, 1).transpose(0, 2).contiguous()
        img = img.float().div(255) if self.div else img.float()
        return img

