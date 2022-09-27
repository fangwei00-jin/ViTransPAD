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
        self.save_dir = config['test']['save_dir']
        self.w = config['data_loader']['w']
        self.h = config['data_loader']['h']
        if debug:
            self.config['train']['trainer']['save_freq'] = 5
            self.config['train']['trainer']['valid_freq'] = 5
            self.config['train']['trainer']['iterations'] = 5

        # setup data set and data loader
        self.train_dataset = Dataset(config, split='train',  debug=debug)
        self.test_dataset = Dataset(config, split='train',  debug=debug)
        self.train_sampler = None
        self.test_sampler = None
        self.train_args = config['train']['trainer']
        if config['distributed']:
            self.train_sampler = DistributedSampler(
                self.train_dataset,
                num_replicas=config['world_size'], 
                rank=config['global_rank'])
            self.test_sampler = DistributedSampler(
                self.test_dataset,
                num_replicas=config['world_size'],
                rank=config['global_rank'])
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.train_args['batch_size'] // config['world_size'],
            shuffle=(self.train_sampler is None), 
            num_workers=self.train_args['num_workers'],
            sampler=self.train_sampler)
        self.test_loader = DataLoader(
            self.test_dataset,
            batch_size=self.train_args['batch_size'] // config['world_size'],
            shuffle=(self.test_sampler is None),
            num_workers=self.train_args['num_workers'],
            sampler=self.test_sampler)

        # set loss functions 
        self.adversarial_loss = AdversarialLoss(type=self.config['train']['losses']['GAN_LOSS'])
        self.adversarial_loss = self.adversarial_loss.to(self.config['device'])
        self.l1_loss = nn.L1Loss()
        self.bce_loss = nn.CrossEntropyLoss()

        # setup models including generator and discriminator
        net = importlib.import_module('model.'+config['data_loader']['model'])
        #self.netG = net.InpaintGenerator()
        self.netG = net.InpaintGenerator(self.config['data_loader']['Transformer_layers'],
                                         self.config['data_loader']['Transformer_heads'],
                                         self.config['data_loader']['channel'],
                                         self.config['data_loader']['patchsize'])
        self.netG = self.netG.to(self.config['device'])
        self.netD = net.Discriminator(
            in_channels=3, use_sigmoid=config['train']['losses']['GAN_LOSS'] != 'hinge')
        self.netD = self.netD.to(self.config['device'])
        self.optimG = torch.optim.Adam(
            self.netG.parameters(), 
            lr=config['train']['trainer']['lr'],
            betas=(self.config['train']['trainer']['beta1'], self.config['train']['trainer']['beta2']))
        for name, p in self.netG.named_parameters():
            if p.requires_grad == True:
                print (name)

        self.optimD = torch.optim.Adam(
            self.netD.parameters(), 
            lr=config['train']['trainer']['lr'],
            betas=(self.config['train']['trainer']['beta1'], self.config['train']['trainer']['beta2']))
        self.load()

        # ## evaluate
        # self.evaluate = Evaluate(config,'train', self.netG)
        config['distributed'] = False
        if config['distributed']:
            self.netG = DDP(
                self.netG, 
                device_ids=[self.config['local_rank']],
                output_device=self.config['local_rank'],
                broadcast_buffers=True, 
                find_unused_parameters=True)
            self.netD = DDP(
                self.netD, 
                device_ids=[self.config['local_rank']], 
                output_device=self.config['local_rank'],
                broadcast_buffers=True, 
                find_unused_parameters=True)
        # else:
        #     self.netG = nn.DataParallel(self.netG.cuda(), device_ids=[1,2,3])  # select GPU 1, 2, 3
        #     self.netD = nn.DataParallel(self.netD.cuda(), device_ids=[1,2,3])  # select GPU 1, 2, 3

        # set summary writer
        self.dis_writer = None
        self.gen_writer = None
        self.summary = {}
        if self.config['global_rank'] == 0 or (not config['distributed']):
            self.dis_writer = SummaryWriter(
                os.path.join(config['test']['save_dir'], 'dis'))
            self.gen_writer = SummaryWriter(
                os.path.join(config['test']['save_dir'], 'gen'))



    # get current learning rate
    def get_lr(self):
        return self.optimG.param_groups[0]['lr']

     # learning rate scheduler, step
    # def adjust_learning_rate(self):
    #     # decay = 0.1**(min(self.iteration,
    #     #                   self.config['trainer']['niter_steady']) // self.config['trainer']['niter'])
    #     decay = 0.5 ** (min(self.iteration, self.config['trainer']['niter_steady'])
    #                     // self.config['trainer']['niter'])
    #     new_lr = self.config['trainer']['lr'] * decay
    #     if new_lr != self.get_lr():
    #         for param_group in self.optimG.param_groups:
    #             param_group['lr'] = new_lr
    #         for param_group in self.optimD.param_groups:
    #             param_group['lr'] = new_lr
    def adjust_learning_rate(self):
        decay = 0.5
        new_lr = self.get_lr() * decay
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

    # load netG
    def load(self):
        if os.path.isfile(self.config['test']['ckpt']):
            gen_path = self.config['test']['ckpt']
            data = torch.load(gen_path, map_location=self.config['device'])
            self.netG.load_state_dict(data['netG'])
            if self.config['global_rank'] == 0:
                print('Loading pretrained model from {}...'.format(gen_path))
        # model_path = self.config['train']['save_dir']
        #
        # ## load the pretrained model
        # if os.path.isfile(self.config['train']['pretrained_model']):
        #     gen_path = self.config['train']['pretrained_model']
        #     data = torch.load(gen_path, map_location=self.config['device'])
        #     self.netG.load_state_dict(data['netG'])
        #     if self.config['global_rank'] == 0:
        #         print('Loading pretrained model from {}...'.format(gen_path))
        # else:
        #
        #     if os.path.isfile(os.path.join(model_path, 'latest.ckpt')):
        #         latest_epoch = open(os.path.join(
        #             model_path, 'latest.ckpt'), 'r').read().splitlines()[-1]
        #     else:
        #         ckpts = [os.path.basename(i).split('.pth')[0] for i in glob.glob(
        #             os.path.join(model_path, '*.pth'))]
        #         ckpts.sort()
        #         latest_epoch = str.split(ckpts[-1], '_')[-1] if len(ckpts) > 0 else None
        #     if latest_epoch is not None:
        #
        #         gen_path = os.path.join(
        #             model_path, 'gen_{}.pth'.format(str(latest_epoch).zfill(5)))
        #         dis_path = os.path.join(
        #             model_path, 'dis_{}.pth'.format(str(latest_epoch).zfill(5)))
        #         opt_path = os.path.join(
        #             model_path, 'opt_{}.pth'.format(str(latest_epoch).zfill(5)))
        #
        #         if self.config['global_rank'] == 0:
        #             print('Loading model from {}...'.format(gen_path))
        #         data = torch.load(gen_path, map_location=self.config['device'])
        #         self.netG.load_state_dict(data['netG'])
        #         data = torch.load(dis_path, map_location=self.config['device'])
        #         self.netD.load_state_dict(data['netD'])
        #         data = torch.load(opt_path, map_location=self.config['device'])
        #         self.optimG.load_state_dict(data['optimG'])
        #         self.optimD.load_state_dict(data['optimD'])
        #         self.epoch = data['epoch']
        #         self.iteration = data['iteration']
        #     else:
        #         if self.config['global_rank'] == 0:
        #             print(
        #                 'Warnning: There is no trained model found. An initialized model will be used.')

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
                
            ## delete the old opt_*.pth and dis_*.pth only save the latest dis.pth and opt.pth
            os.system('rm {}'.format(os.path.join(self.config['save_dir'],'opt_*.pth')) )
            os.system('rm {}'.format(os.path.join(self.config['save_dir'],'dis_*.pth')) )

            torch.save({'netG': netG.state_dict()}, gen_path)
            torch.save({'netD': netD.state_dict()}, dis_path)
            torch.save({'epoch': self.epoch,
                        'iteration': self.iteration,
                        'optimG': self.optimG.state_dict(),
                        'optimD': self.optimD.state_dict()}, opt_path)
            os.system('echo {} > {}'.format(str(it).zfill(5),
                                            os.path.join(self.config['train']['save_dir'], 'latest.ckpt')))



    # train entry
    def train(self):
        pbar = range(int(self.train_args['iterations']))
        if self.config['global_rank'] == 0:
            pbar = tqdm(pbar, initial=self.iteration, dynamic_ncols=True, smoothing=0.01)

        #while True:
        self.epoch += 1
        if self.config['distributed']:
            self.test_sampler.set_epoch(self.epoch)

        self._train_epoch(pbar)
        # if self.iteration > self.train_args['iterations']:
        #     break
        ACC_best = 0.0
        APCER_best = 1.0
        BPCER_best = 1.0
        ACER_best = 1.0

        return ACC_best, APCER_best, BPCER_best, ACER_best



    # process input and calculate loss every training epoch
    def _train_epoch(self, pbar):
        tt = [0.0, 0.0]
        device = self.config['device']
        # self.netG.train()
        # self.netD.train()
        with torch.no_grad():

            for batch_idx, batch  in enumerate(self.train_loader):
                self.netG.eval()
                print(batch_idx)
                # ##debug
                # if batch_idx > 3:
                #     break
                #self.adjust_learning_rate()
                self.iteration += 1

                frames, masks, spoofing_labels = batch['frame_tensors'].to(device), batch['mask_tensors'].to(device),  batch['spoofing_labels']
                b, t, c, h, w = frames.size()
                masked_frame = (frames * (1 - masks).float())
                #y_enc = self.netG(masked_frame, masks)
                y_enc = self.netG.infer(masked_frame, masks)



                ### transformer binary classification entropy loss (Transformer BCE)
                images_labels = np.array([t*[label] for label in spoofing_labels])
                images_labels = torch.from_numpy(images_labels.flatten())
                images_labels = images_labels.to(device)
                #loss_softmax_enc = self.bce_loss(y_enc, images_labels)
                # loss_softmax_depth = self.bce_loss(y_dec, images_labels)
                # self.add_summary(
                #     self.gen_writer, 'loss/loss_softmax_enc', loss_softmax_enc.item()
                # )
                # self.add_summary(
                #     self.gen_writer, 'loss/loss_softmax_depth', loss_softmax_depth.item()
                # )

                #bp() ## debug
                gen_loss = 0
                dis_loss = 0

                # gen_loss += 1.0*loss_softmax_enc + 1.0*loss_softmax_depth
                #gen_loss += 1.0*loss_softmax_enc

                # self.optimG.zero_grad()
                # gen_loss.backward()
                # self.optimG.step()

                # console logs
                # if self.config['global_rank'] == 0:
                #     pbar.update(1)
                #     pbar.set_description((
                #         f"d: {dis_loss.item():.3f}; g: {gan_loss.item():.3f};"
                #         f"valid: {valid_loss.item():.3f}")
                #     )
                # if self.config['global_rank'] == 0:
                #     pbar.update(1)
                #     pbar.set_description((
                #         f"d: {gen_loss.item():.3f}; ")
                #     )
                #
                #
                # if self.config['global_rank'] == 0:
                #     tt[self.iteration%2] = time.time()
                #     print('Train Epoch: {}/{} [ {}/{} batch)] time:{:.2f} \ttransformer_softmax_loss: {:.6f} lr: {:.6f}'.format(
                #             int(self.epoch), int(self.train_args['iterations']/len(self.train_loader)), (batch_idx), len(self.train_loader),
                #              abs(tt[-1] - tt[-2]), gen_loss.item(), self.get_lr()))
                #     print("spoofing labels of videos in batch:", end="")
                #     for spoofing_label in batch['spoofing_labels']:
                #         print("%d "%spoofing_label.item(),end="")
                #     print("length batch['spoofing_labels'] : %d"%len(batch['spoofing_labels']))
                #     print('\n')

        # saving models
        if self.epoch % 5 == 0:
            self.save(int(self.epoch))

