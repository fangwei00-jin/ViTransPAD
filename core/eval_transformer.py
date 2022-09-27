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
from core.dataset_video import Dataset
#from core.SiW import SiW as Dataset
from core.loss import AdversarialLoss
#from core.Evaluate import Evaluate
from pdb import set_trace as bp

class Test():
    def __init__(self, config, debug=False):
        self.config = config
        self.epoch = 0
        self.iteration = 0
        self.save_dir = config['save_dir']
        self.w = config['data_loader']['w']
        self.h = config['data_loader']['h']


        # setup data set and data loader
        self.test_dataset = Dataset(config, split='test',  debug=debug)
        self.test_sampler = None
        self.train_args = config['train']['trainer']
        if config['distributed']:
            self.test_sampler = DistributedSampler(
                self.test_dataset,
                num_replicas=config['world_size'],
                rank=config['global_rank'])
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

        #self.load()
        self.model_dict = self.netG.state_dict()
        self.model = self.netG

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
                os.path.join(config['save_dir'], 'dis'))
            self.gen_writer = SummaryWriter(
                os.path.join(config['save_dir'], 'gen'))



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
    def load(self, pretrained_model):
        ## load the pretrained model
        if os.path.isfile(pretrained_model):
            data = torch.load(pretrained_model, map_location=self.config['device'])
            self.model_dict.update(data['netG']) ## update current model parameters
            self.model.load_state_dict(self.model_dict)## don't forget to load the new parameters in model!!
            if self.config['global_rank'] == 0:
                print('Loading pretrained model from {}...'.format(pretrained_model))

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
                                            os.path.join(self.config['save_dir'], 'latest.ckpt')))



    # train entry
    def test(self):
        pbar = range(int(self.train_args['iterations']))
        if self.config['global_rank'] == 0:
            pbar = tqdm(pbar, initial=self.iteration, dynamic_ncols=True, smoothing=0.01)

        self.epoch += 1
        if self.config['distributed']:
            self.test_sampler.set_epoch(self.epoch)

        ACC_best = 0.0
        APCER_best = 1.0
        BPCER_best = 1.0
        ACER_best = 1.0
        model_best = ""
        with open(os.path.join(self.save_dir, "evaluation.txt"),"w") as f:
            f.write("ACC   APCER   BPCER   ACER  Model\n")
            if os.path.isdir(self.config['test']['ckpt']):
                models = glob.glob(os.path.join(self.config['test']['ckpt'], "gen_*.pth"))
                models.sort()
                for ep, model in enumerate(models):
                    self.load(model)
                    eval_results = [ACC_best, APCER_best, BPCER_best, ACER_best, model_best, model, len(models), ep]
                    y_labels, gt_labels = self._test_epoch(pbar,eval_results)

                    ACC, APCER, BPCER, ACER = self.performance(y_labels, gt_labels)
                    if ACER < ACER_best:
                        ACC_best = ACC
                        APCER_best = APCER
                        BPCER_best = BPCER
                        ACER_best = ACER
                        model_best = model
                    if self.config['global_rank'] == 0:
                        print("ACC=%.4f, APCER=%.4f, BPCER=%.4f, ACER=%.4f  Model:%s \n"%(ACC, APCER, BPCER, ACER, model))
                        f.write("ACC=%.4f, APCER=%.4f, BPCER=%.4f, ACER=%.4f  Model:%s\n"%(ACC, APCER, BPCER, ACER, model))

                f.write("Best model============> \n")
                f.write("ACC=%.4f, APCER=%.4f, BPCER=%.4f, ACER=%.4f Model:%s\n" % (ACC_best, APCER_best, BPCER_best, ACER_best, model_best))

        return ACC_best, APCER_best, BPCER_best, ACER_best


    def performance (self, y_labels, labels, threshold=0.5):
        idx_type1 = [i for i, y in enumerate(y_labels) if y < threshold and labels[i] == 1] ## false negative
        idx_type2 = [i for i, y in enumerate(y_labels) if y > threshold and labels[i] == 0] ## false positive
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




    # process input and calculate loss every training epoch
    def _test_epoch(self, pbar, eval_results):
        tt = [0.0, 0.0]
        eval_epochs = eval_results[6]
        ep = eval_results[7]
        device = self.config['device']
        self.netG.eval()
        self.netD.eval()
        Y_labels = []
        GT_labels = []

        #############################################################################
        #######       torch.no_grad() very important for test !!!!!!!        ########
        ####### otherwise the variable with grad attribute will accumulate   ########
        #######  the gradient information resulting the GPU memory explosion. #######
        ####### model.eval() only disables the batch_norm() and dropout()!   ########
        ####### model.eval() will not stop the gradient calculation.         ########
        ####### During the training, loss.backwards() will release the       ########
        ####### calculation graph which also releases all the variables to   ########
        ####### allow the GPU memory won't be all occupied!!!                ########
        #############################################################################
        with torch.no_grad():
            for batch_idx, batch  in enumerate(self.test_loader):
                self.iteration += 1

                frames, masks, spoofing_labels = batch['frame_tensors'].to(device), batch['mask_tensors'].to(device),  batch['spoofing_labels']
                b, t, c, h, w = frames.size()
                masked_frame = (frames * (1 - masks).float())
                y_enc = self.netG(masked_frame, masks)
                y_labels = [np.argmax(y) for y in y_enc.detach().cpu().tolist()]
                Y_labels += y_labels
                images_labels = np.array([t*[label] for label in spoofing_labels])
                images_labels_flat = [item for sublist in images_labels for item in sublist]
                GT_labels += images_labels_flat

                # ##debug
                # ACC, APCER, BPCER, ACER = self.performance(Y_labels, GT_labels)

                ### transformer binary classification entropy loss (Transformer BCE)
                images_labels = torch.from_numpy(images_labels.flatten())
                images_labels = images_labels.to(device)
                loss_softmax_enc = self.bce_loss(y_enc, images_labels)
                # loss_softmax_depth = self.bce_loss(y_dec, images_labels)


                #bp() ## debug
                gen_loss = 0
                gen_loss += 1.0*loss_softmax_enc

                if self.config['global_rank'] == 0:
                    pbar.update(1)
                    pbar.set_description((
                        f"d: {gen_loss.item():.3f}; ")
                    )


                if self.config['global_rank'] == 0:
                    tt[self.iteration%2] = time.time()
                    print("Evaluating Model:%s \n" % (eval_results[5]))
                    print("ACC_best=%.4f, APCER_best=%.4f, BPCER_best=%.4f, ACER_best=%.4f  Model_best:%s \n" % (eval_results[0], eval_results[1], eval_results[2], eval_results[3], str.split(eval_results[4], '/')[-1]))
                    print('Evaluating Epoch: {}/{} [ {}/{} batch)] time:{:.2f} \ttransformer_softmax_loss: {:.6f} lr: {:.6f}'.format(
                            ep, eval_epochs, (batch_idx), len(self.test_loader),
                             abs(tt[-1] - tt[-2]), gen_loss.item(), self.get_lr()))
                    print("spoofing labels of videos in batch:", end="")
                    for spoofing_label in batch['spoofing_labels']:
                        print("%d "%spoofing_label.item(),end="")
                    print("length batch['spoofing_labels'] : %d"%len(batch['spoofing_labels']))
                    print('\n')

                # ##debug
                # break

        return Y_labels, GT_labels