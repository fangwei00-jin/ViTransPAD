import os
import cv2
import time
import math
import glob
from tqdm import tqdm
import shutil
import importlib
import datetime
import numpy as np
from PIL import Image
from math import log10

from functools import partial
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from tensorboardX import SummaryWriter
from torchvision.utils import make_grid, save_image
import torch.distributed as dist
from sklearn.metrics import roc_curve, auc
from core.dataset import Dataset
#from core.SiW import SiW as Dataset
from core.loss import AdversarialLoss
from core.Evaluate import Evaluate
from pdb import set_trace as bp

class Trainer():
    def __init__(self, config, debug=False):
        self.config = config
        self.epoch = 0
        self.iteration = 0
        self.eval_epoch_num = config['trainer']['eval_epoch_num']
        self.save_dir = config['save_dir']
        self.w = config['data_loader']['w']
        self.h = config['data_loader']['h']
        if debug:
            self.config['trainer']['save_freq'] = 5
            self.config['trainer']['valid_freq'] = 5
            self.config['trainer']['iterations'] = 5

        # setup data set and data loader
        self.train_dataset = Dataset(config, split='train',  debug=debug)
        # self.test_dataset = Dataset(config['data_loader'], config['dataset'], config['test']['face_scale'], split='test',  debug=debug)
        self.train_sampler = None
        # self.test_sampler = None
        self.train_args = config['trainer']
        if config['distributed']:
            self.train_sampler = DistributedSampler(
                self.train_dataset,
                num_replicas=config['world_size'], 
                rank=config['global_rank'])
            # self.test_sampler = DistributedSampler(
            #     self.test_dataset,
            #     num_replicas=config['world_size'],
            #     rank=config['global_rank'])
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.train_args['batch_size'] // config['world_size'],
            shuffle=(self.train_sampler is None), 
            num_workers=self.train_args['num_workers'],
            sampler=self.train_sampler)
        # self.test_loader = DataLoader(
        #     self.test_dataset,
        #     batch_size=self.train_args['batch_size'] // config['world_size'],
        #     shuffle=(self.test_sampler is None),
        #     num_workers=self.train_args['num_workers'],
        #     sampler=self.test_sampler)

        # set loss functions 
        self.adversarial_loss = AdversarialLoss(type=self.config['losses']['GAN_LOSS'])
        self.adversarial_loss = self.adversarial_loss.to(self.config['device'])
        self.l1_loss = nn.L1Loss()

        # setup models including generator and discriminator
        net = importlib.import_module('model.'+config['trainer']['model'])
        #self.netG = net.InpaintGenerator()
        self.netG = net.InpaintGenerator(self.config['trainer']['Transformer_layers'], self.config['trainer']['Transformer_heads'], self.config['trainer']['channel'])
        self.netG = self.netG.to(self.config['device'])
        self.netD = net.Discriminator(
            in_channels=3, use_sigmoid=config['losses']['GAN_LOSS'] != 'hinge')
        self.netD = self.netD.to(self.config['device'])
        self.optimG = torch.optim.Adam(
            self.netG.parameters(), 
            lr=config['trainer']['lr'],
            betas=(self.config['trainer']['beta1'], self.config['trainer']['beta2']))
        self.optimD = torch.optim.Adam(
            self.netD.parameters(), 
            lr=config['trainer']['lr'],
            betas=(self.config['trainer']['beta1'], self.config['trainer']['beta2']))
        self.load()

        ## evaluate
        self.evaluate = Evaluate(config,'train', self.netG)

        if config['distributed']:
            self.netG = DDP(
                self.netG, 
                device_ids=[self.config['local_rank']], 
                output_device=self.config['local_rank'],
                broadcast_buffers=True, 
                find_unused_parameters=False)
            self.netD = DDP(
                self.netD, 
                device_ids=[self.config['local_rank']], 
                output_device=self.config['local_rank'],
                broadcast_buffers=True, 
                find_unused_parameters=False)
        # else:
        #     self.netG = nn.DataParallel(self.netG.cuda(), device_ids=[1,2,3])  # select GPU 1, 2, 3
        #     self.netD = nn.DataParallel(self.netD.cuda(), device_ids=[1,2,3])  # select GPU 1, 2, 3

        # set summary writer
        self.dis_writer = None
        self.gen_writer = None
        self.summary = {}
        if self.config['global_rank'] == 0 or (not config['distributed']):
            self.dis_writer = SummaryWriter(
                os.path.join(config['save_dir'], 'dis'))
            self.gen_writer = SummaryWriter(
                os.path.join(config['save_dir'], 'gen'))



    # get current learning rate
    def get_lr(self):
        return self.optimG.param_groups[0]['lr']

     # learning rate scheduler, step
    def adjust_learning_rate(self):
        decay = 0.1**(min(self.iteration,
                          self.config['trainer']['niter_steady']) // self.config['trainer']['niter'])
        new_lr = self.config['trainer']['lr'] * decay
        if new_lr != self.get_lr():
            for param_group in self.optimG.param_groups:
                param_group['lr'] = new_lr
            for param_group in self.optimD.param_groups:
                param_group['lr'] = new_lr

    # add summary
    def add_summary(self, writer, name, val):
        if name not in self.summary:
            self.summary[name] = 0
        self.summary[name] += val
        if writer is not None and self.iteration % 100 == 0:
            writer.add_scalar(name, self.summary[name]/100, self.iteration)
            self.summary[name] = 0

    # load netG and netD
    def load(self):
        model_path = self.config['save_dir']
        if os.path.isfile(os.path.join(model_path, 'latest.ckpt')):
            latest_epoch = open(os.path.join(
                model_path, 'latest.ckpt'), 'r').read().splitlines()[-1]
        else:
            ckpts = [os.path.basename(i).split('.pth')[0] for i in glob.glob(
                os.path.join(model_path, '*.pth'))]
            ckpts.sort()
            latest_epoch = str.split(ckpts[-1], '_')[-1] if len(ckpts) > 0 else None
        if latest_epoch is not None:

            gen_path = os.path.join(
                model_path, 'gen_{}.pth'.format(str(latest_epoch).zfill(5)))
            dis_path = os.path.join(
                model_path, 'dis_{}.pth'.format(str(latest_epoch).zfill(5)))
            opt_path = os.path.join(
                model_path, 'opt_{}.pth'.format(str(latest_epoch).zfill(5)))
            if self.config['global_rank'] == 0:
                print('Loading model from {}...'.format(gen_path))
            data = torch.load(gen_path, map_location=self.config['device'])
            self.netG.load_state_dict(data['netG'])
            data = torch.load(dis_path, map_location=self.config['device'])
            self.netD.load_state_dict(data['netD'])
            data = torch.load(opt_path, map_location=self.config['device'])
            self.optimG.load_state_dict(data['optimG'])
            self.optimD.load_state_dict(data['optimD'])
            self.epoch = data['epoch']
            self.iteration = data['iteration']
        else:
            if self.config['global_rank'] == 0:
                print(
                    'Warnning: There is no trained model found. An initialized model will be used.')

    # save parameters every eval_epoch
    def save(self, it):
        if self.config['global_rank'] == 0:
            gen_path = os.path.join(
                self.config['save_dir'], 'gen_{}.pth'.format(str(it).zfill(5)))
            dis_path = os.path.join(
                self.config['save_dir'], 'dis_{}.pth'.format(str(it).zfill(5)))
            opt_path = os.path.join(
                self.config['save_dir'], 'opt_{}.pth'.format(str(it).zfill(5)))
            print('\nsaving model to {} ...'.format(gen_path))
            if isinstance(self.netG, torch.nn.DataParallel) or isinstance(self.netG, DDP):
                netG = self.netG.module
                netD = self.netD.module
            else:
                netG = self.netG
                netD = self.netD
            torch.save({'netG': netG.state_dict()}, gen_path)
            torch.save({'netD': netD.state_dict()}, dis_path)
            torch.save({'epoch': self.epoch,
                        'iteration': self.iteration,
                        'optimG': self.optimG.state_dict(),
                        'optimD': self.optimD.state_dict()}, opt_path)
            os.system('echo {} > {}'.format(str(it).zfill(5),
                                            os.path.join(self.config['save_dir'], 'latest.ckpt')))



    # train entry
    def train(self):
        pbar = range(int(self.train_args['iterations']))
        if self.config['global_rank'] == 0:
            pbar = tqdm(pbar, initial=self.iteration, dynamic_ncols=True, smoothing=0.01)

        ACC_all = []
        APCER_all = []
        BPCER_all = []
        ACER_all = []
        threshold_all = []

        while True:
            self.epoch += 1
            if self.config['distributed']:
                self.train_sampler.set_epoch(self.epoch)

            self._train_epoch(pbar)
            if self.iteration > self.train_args['iterations']:
                break

            # if self.epoch%self.eval_epoch_num == 0   and self.config['global_rank'] == 0:
            #     device = "cuda:%d"%self.config['local_rank']
            #     ACC, APCER, BPCER, ACER, threshold, fprs, tprs, thresholds, acc_list, thresholds_test = self.evaluate.evaluate(device, self.epoch)
            #     ACC_all.append(ACC)
            #     APCER_all.append(APCER)
            #     BPCER_all.append(BPCER)
            #     ACER_all.append(ACER)
            #     threshold_all.append(threshold)
            #     print('threshold= %.4f, ACC= %.4f, APCER= %.4f, BPCER= %.4f, ACER= %.4f' % (
            #         threshold, ACC, APCER, BPCER, ACER))
            #     with open(os.path.join(self.save_dir, 'ep_%03d'%self.epoch, 'log.txt'), 'w') as f:
            #         f.write('threshold= %.4f, ACC= %.4f, APCER= %.4f, BPCER= %.4f, ACER= %.4f\n' % (
            #             threshold, ACC, APCER, BPCER, ACER))
            #     with open(os.path.join(self.save_dir, 'ep_%03d'%self.epoch, 'val_fpr.txt'), 'w') as f:
            #         f.write('fps\n')
            #         for i, fpr in enumerate(fprs):
            #             f.write('%f\n' % (fpr))
            #     with open(os.path.join(self.save_dir, 'ep_%03d'%self.epoch, 'val_tpr.txt'), 'w') as f:
            #         f.write('tprs\n')
            #         for i, fpr in enumerate(fprs):
            #             f.write('%f\n' % (tprs[i]))
            #     with open(os.path.join(self.save_dir, 'ep_%03d'%self.epoch,  'val_threholds.txt'), 'w') as f:
            #         f.write('thresholds\n')
            #         for i, fpr in enumerate(fprs):
            #             f.write('%f\n' % (thresholds[i]))
            #     with open(os.path.join(self.save_dir, 'ep_%03d'%self.epoch,  'acc_test.txt'), 'w') as f:
            #         f.write('acc\n')
            #         for i, acc in enumerate(acc_list):
            #             f.write('%f\n' % (acc))
            #     with open(os.path.join(self.save_dir, 'ep_%03d'%self.epoch,  'threholds_test.txt'), 'w') as f:
            #         f.write('thresholds_test\n')
            #         for i, acc in enumerate(acc_list):
            #             f.write('%f\n' % (thresholds_test[i]))

        # print('\nEnd training....')
        # with open(os.path.join(self.save_dir, 'ACC_all.txt'), 'w') as f:
        #     f.write('ACC_all\n')
        #     for i, acc in enumerate(ACC_all):
        #         f.write('%f\n' % (acc))
        # with open(os.path.join(self.save_dir, 'APCER_all.txt'), 'w') as f:
        #     f.write('APCER_all\n')
        #     for i, apcer in enumerate(APCER_all):
        #         f.write('%f\n' % (apcer))
        # with open(os.path.join(self.save_dir, 'BPCER_all.txt'), 'w') as f:
        #     f.write('BPCER_all\n')
        #     for i, bpcer in enumerate(BPCER_all):
        #         f.write('%f\n' % (bpcer))
        # with open(os.path.join(self.save_dir, 'ACER_all.txt'), 'w') as f:
        #     f.write('ACER_all\n')
        #     for i, acer in enumerate(ACER_all):
        #         f.write('%f\n' % (acer))
        # with open(os.path.join(self.save_dir, 'threshold_all.txt'), 'w') as f:
        #     f.write('threshold_all\n')
        #     for i, threshold in enumerate(threshold_all):
        #         f.write('%f\n' % (threshold))





    # process input and calculate loss every training epoch
    def _train_epoch(self, pbar):
        tt = [0.0, 0.0]
        device = self.config['device']
        self.netG.train()
        self.netD.train()

        for batch_idx, batch  in enumerate(self.train_loader):
            # ##debug
            # if batch_idx > 3:
            #     break
            self.adjust_learning_rate()
            self.iteration += 1

            frames, masks, depth_maps = batch['frame_tensors'].to(device), batch['mask_tensors'].to(device), batch['map_tensors'].to(device)
            b, t, c, h, w = frames.size()
            masked_frame = (frames * (1 - masks).float())
            pred_depth_maps = self.netG(masked_frame, masks)
            # frames = frames.view(b*t, c, h, w)
            masks = masks.view(b*t, 1, h, w)
            depth_maps = depth_maps.view(b*t, c, h, w)
            # comp_img = frames*(1.-masks) + masks*pred_img
            
            #bp() ## debug

            gen_loss = 0
            dis_loss = 0

            # discriminator adversarial loss
            # real_vid_feat = self.netD(frames)
            real_vid_feat = self.netD(depth_maps)
            #fake_vid_feat = self.netD(comp_img.detach())
            fake_vid_feat = self.netD(pred_depth_maps.detach())
            ## first True/Fake means inputing real/generated fake image feature, second True/False means for training Discriminator/Generator
            dis_real_loss = self.adversarial_loss(real_vid_feat, True, True)
            dis_fake_loss = self.adversarial_loss(fake_vid_feat, False, True)
            dis_loss += (dis_real_loss + dis_fake_loss) / 2
            self.add_summary(
                self.dis_writer, 'loss/dis_vid_fake', dis_fake_loss.item())
            self.add_summary(
                self.dis_writer, 'loss/dis_vid_real', dis_real_loss.item())
            self.optimD.zero_grad()
            dis_loss.backward()
            self.optimD.step()

            # generator adversarial loss
            # gen_vid_feat = self.netD(comp_img)
            gen_vid_feat = self.netD(pred_depth_maps)
            ## The second Fake means not for trainning Generator but not for trainining Discriminator, now the first Ture/False has no use.
            gan_loss = self.adversarial_loss(gen_vid_feat, True, False)
            gan_loss = gan_loss * self.config['losses']['adversarial_weight']
            gen_loss += gan_loss
            self.add_summary(
                self.gen_writer, 'loss/gan_loss', gan_loss.item())

            # generator l1 loss
            # hole_loss = self.l1_loss(pred_img*masks, frames*masks)
            # hole_loss = hole_loss / torch.mean(masks) * self.config['losses']['hole_weight']
            # gen_loss += hole_loss
            # self.add_summary(
            #     self.gen_writer, 'loss/hole_loss', hole_loss.item())

            valid_loss = self.l1_loss(pred_depth_maps*(1-masks), depth_maps*(1-masks))
            valid_loss = valid_loss / torch.mean(1-masks) * self.config['losses']['valid_weight']
            gen_loss += valid_loss 
            self.add_summary(
                self.gen_writer, 'loss/valid_loss', valid_loss.item())
            
            self.optimG.zero_grad()
            gen_loss.backward()
            self.optimG.step()

            # console logs
            if self.config['global_rank'] == 0:
                pbar.update(1)
                pbar.set_description((
                    f"d: {dis_loss.item():.3f}; g: {gan_loss.item():.3f};"
                    f"valid: {valid_loss.item():.3f}")
                )

            # # saving models
            # if self.iteration % self.train_args['save_freq'] == 0:
            #     self.save(int(self.iteration//self.train_args['save_freq']))
            # if self.iteration > self.train_args['iterations']:
            #     break


            frames_original = batch['frames']
            masked_frames_original = batch['masks']
            depth_maps_original = batch['maps']

            if self.config['global_rank'] == 0:
                # save generated depth map:
                k = len(self.train_loader)
                if self.iteration%k == 0: # == 0:
                    depth_map_dir = os.path.join(self.config['save_dir'], 'depth_maps_%04d'%(self.iteration//k))
                    if not os.path.isdir(depth_map_dir):
                        os.mkdir(depth_map_dir)
                    pred_depth_maps_detach = pred_depth_maps.detach()
                    pred_depth_maps_detach = (pred_depth_maps_detach + 1) / 2
                    pred_depth_maps_detach = pred_depth_maps_detach.cpu().permute(0, 2, 3, 1).numpy() * 255
                    frmcnt = 0
                    for i in range(len(frames_original)):
                        for j in range(len(frames_original[i])):
                            # cv2.imwrite(os.path.join(depth_map_dir, '%03d_frame.jpg' % i), np.array(frames_original[i][j]))
                            cv2.imwrite(os.path.join(depth_map_dir, '%03d_masked_frame.jpg' % frmcnt), np.array(frames_original[i][j]*(1-cv2.cvtColor(np.array(masked_frames_original[i][j]), cv2.COLOR_GRAY2BGR))))
                            cv2.imwrite(os.path.join(depth_map_dir, '%03d_depth_gt.jpg' % frmcnt), np.array(depth_maps_original[i][j]))
                            frmcnt += 1
                    for i, pred_depth_map in enumerate(pred_depth_maps_detach):
                        cv2.imwrite(os.path.join(depth_map_dir,'%03d_depth_pred.jpg'%i), pred_depth_map)

                ## write predicted depth_map to tensorboard
                write_summary_freq = 10
                if self.iteration % write_summary_freq == 0:
                    iteration = self.iteration
                    n = depth_maps_original.shape[0]
                    frames_batch = np.zeros((n, frames_original.shape[2], frames_original.shape[3], frames_original.shape[4]), dtype=np.uint8)
                    map_batch = np.zeros((n, depth_maps_original.shape[2], depth_maps_original.shape[3], depth_maps_original.shape[4]), dtype=np.uint8)
                    pred_map_batch_show = np.zeros((n, depth_maps_original.shape[2], depth_maps_original.shape[3], depth_maps_original.shape[4]), dtype=np.uint8)
                    pred_depth_maps_detach = pred_depth_maps.detach()
                    pred_depth_maps_detach = (pred_depth_maps_detach + 1) / 2
                    pred_depth_maps_detach = pred_depth_maps_detach.cpu().permute(0, 2, 3, 1).numpy() * 255
                    ## write to tensorboard
                    for i in range(frames_original.shape[0]):
                            frames_batch[i]  = cv2.cvtColor(np.array(frames_original[i][0]), cv2.COLOR_BGR2RGB)
                            map_batch[i]  = depth_maps_original[i][0]
                            pred_map_batch_show[i]  = pred_depth_maps_detach[i*t]

                    # map_batch_show[:n, :, :, :] = map_batch
                    # map_batch_show[n:, :, :, :] = pred_depth_mapspred_depth_maps_detach
                    self.gen_writer.add_images('images', frames_batch, iteration/write_summary_freq, dataformats='NHWC')
                    self.gen_writer.add_images('map_x_gt', map_batch, iteration/write_summary_freq, dataformats='NHWC')
                    self.gen_writer.add_images('map_x_estimate', pred_map_batch_show, iteration/write_summary_freq, dataformats='NHWC')
                    #self.gen_writer.flush()

                ## print loss information
                if self.config['global_rank'] == 0:
                    tt[self.iteration%2] = time.time()
                    print('Train Epoch: {}/{} [ {}/{} batch)] time:{:.2f} \tdepth_map_gen_loss: {:.6f}  dis_loss: {:.6f}'.format(
                            int(self.epoch), int(self.train_args['iterations']/len(self.train_loader)), (batch_idx), len(self.train_loader),
                             abs(tt[-1] - tt[-2]), valid_loss.item(), dis_loss.item()))
                    print("spoofing labels of videos in batch:", end="")
                    for spoofing_label in batch['spoofing_labels']:
                        print("%d "%spoofing_label.item(),end="")
                    print("length batch['spoofing_labels'] : %d"%len(batch['spoofing_labels']))
                    print('\n')

        # saving models
        self.save(int(self.epoch))

