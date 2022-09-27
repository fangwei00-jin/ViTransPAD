import os
import cv2
import time
import math
import glob
# from tqdm import tqdm
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
# from core.dataset_video import Dataset
# from core.dataset_image import Dataset
# from core.SiW import SiW as Dataset
from core.loss import AdversarialLoss
from pdb import set_trace as bp
import importlib
from core.cosine_annealing_warmup import CosineAnnealingWarmupRestarts
import random
from thop import profile
from copy import deepcopy
#########################################################################
### pip install torchmetrics  #### for distributed evaluation,
### DDP can only synchronize bp of different gpus, but not for forward,
### the distributed forward on each gpu is independant;
#########################################################################
import torchmetrics

## Attention visualisation from https://github.com/jacobgil/pytorch-grad-cam
from pytorch_grad_cam import GradCAM, \
    ScoreCAM, \
    GradCAMPlusPlus, \
    AblationCAM, \
    XGradCAM, \
    EigenCAM, \
    EigenGradCAM, \
    LayerCAM, \
    FullGrad
from pytorch_grad_cam import GuidedBackpropReLUModel
from pytorch_grad_cam.utils.image import show_cam_on_image, preprocess_image


class Trainer():
    def __init__(self, config, debug=False):
        self.config = config
        self.setup_seed(config['train']['seed'])
        mode = config['data_loader']['mode']
        # ########################################################
        # ### using torchmetrics for distributed evaluation #####
        # self.metric = torchmetrics.Accuracy()
        # self.metric.to(self.config['device'])
        # ########################################################
        try:
            dataset_module = importlib.import_module("core." + mode)
            Dataset = dataset_module.Dataset
        except:
            raise ValueError("Input correct mode of input: dataset_image or dataset_video !")
        self.sample_length = config['data_loader']['sample_length']
        self.epoch = 0
        self.epoch_best = 0
        self.ACC_best = 0.0
        self.APCER_best = 1.0
        self.BPCER_best = 1.0
        self.ACER_best = 1.0

        self.iteration = 0
        self.eval_epoch_num = config['train']['trainer']['eval_epoch_num']
        self.save_dir = config['train']['save_dir']
        self.w = config['data_loader']['w']
        self.h = config['data_loader']['h']
        self.classes = self.config['data_loader']['class']
        self.lr = config['train']['trainer']['lr']
        self.update_step = config['train']['trainer']['update_step']
        if debug:
            self.config['train']['trainer']['save_freq'] = 5
            self.config['train']['trainer']['valid_freq'] = 5
            self.config['train']['trainer']['iterations'] = 5

        ############ setup data set and data loader for Meta-learning ###############
        if self.config['train']['meta_learning'] == "":
            raise vaulueError("Meta-learning is not used ! Please set Meta-learning strategy in configuration file!")
        else:
            meta_datasets = str.split(self.config['train']['modality'][0], ';')
            self.train_args = config['train']['trainer']
            self.meta_train_loader_list = []
            self.meta_test_loader_list = []
            for meta_dataset_i in meta_datasets:
                meta_train_dataset_i, meta_test_dataset_i = str.split(meta_dataset_i, ',')

                meta_train_dataset = Dataset(config, [meta_train_dataset_i], split='train',  debug=debug)
                meta_test_dataset = Dataset(config, [meta_test_dataset_i], split='train',  debug=debug)
                self.meta_train_sampler = None
                self.meta_test_sampler = None
                if config['distributed']:
                    self.meta_train_sampler = DistributedSampler(
                        meta_train_dataset,
                        num_replicas=config['world_size'],
                        rank=config['global_rank'])
                    meta_test_sampler = DistributedSampler(
                        meta_test_dataset,
                        num_replicas=config['world_size'],
                        rank=config['global_rank'])
                meta_train_loader = DataLoader(
                    meta_train_dataset,
                    batch_size=self.train_args['batch_size'] // config['world_size'],
                    shuffle=(self.meta_train_sampler is None),
                    num_workers=self.train_args['num_workers'],
                    pin_memory=True,
                    sampler=self.meta_train_sampler)
                meta_test_loader = DataLoader(
                    meta_test_dataset,
                    batch_size=self.train_args['batch_size'] // config['world_size'],
                    shuffle=(self.meta_test_sampler is None),
                    num_workers=self.train_args['num_workers'],
                    pin_memory=True,
                    sampler=self.meta_test_sampler)
                self.meta_train_loader_list.append(meta_train_loader)
                self.meta_test_loader_list.append(meta_test_loader)

            ########## data set and data loader for test in Meta-learning are same as normal case #######
            self.test_dataset = Dataset(config, self.config['test']['test_data'], split='test', debug=debug)
            self.test_sampler = None
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
                pin_memory=True,
                sampler=self.test_sampler)

        # set loss functions
        self.adversarial_loss = AdversarialLoss(type=self.config['train']['losses']['GAN_LOSS'])
        self.adversarial_loss = self.adversarial_loss.to(self.config['device'])
        self.l1_loss = nn.L1Loss()
        self.bce_loss = nn.CrossEntropyLoss()

        # setup models including generator and discriminator
        net = importlib.import_module('model.' + config['data_loader']['model'])
        # self.netG = net.InpaintGenerator()

        self.netG = net.InpaintGenerator(self.config['data_loader']['Transformer_layers'],
                                         self.config['data_loader']['Transformer_heads'],
                                         self.config['data_loader']['channel'],
                                         self.config['data_loader']['patchsize'],
                                         self.config['data_loader']['backbone'],
                                         self.config['data_loader']['featurelayer'],
                                         self.config['data_loader']['class']
                                         )
        #
        # ## sttn_transformer_Efficientnet_FrameAtten
        # self.netG = net.InpaintGenerator(self.config['data_loader']['Transformer_layers'],
        #                                  self.config['data_loader']['Transformer_heads'],
        #                                  self.config['data_loader']['channel'],
        #                                  self.config['data_loader']['patchsize']
        #                                  )

        self.netG = self.netG.to(self.config['device'])
        self.netD = net.Discriminator(
            in_channels=3, use_sigmoid=config['train']['losses']['GAN_LOSS'] != 'hinge')
        self.netD = self.netD.to(self.config['device'])

        if config['train']['trainer']['type'] == "Adam":
            self.optimG = torch.optim.Adam(
                self.netG.parameters(),
                lr=config['train']['trainer']['lr'],
                betas=(self.config['train']['trainer']['beta1'], self.config['train']['trainer']['beta2']))
            self.optimD = torch.optim.Adam(
                self.netD.parameters(),
                lr=config['train']['trainer']['lr'],
                betas=(self.config['train']['trainer']['beta1'], self.config['train']['trainer']['beta2']))
        elif config['train']['trainer']['type'] == "SGD":
            self.optimG = torch.optim.SGD(
                self.netG.parameters(),
                lr=config['train']['trainer']['lr'],
                momentum=0.9,
                weight_decay=0.01)
            self.optimD = torch.optim.SGD(
                self.netD.parameters(),
                lr=config['train']['trainer']['lr'],
                momentum=0.9,
                weight_decay=0.01)
        else:
            raise ValueError("Please select optimizer as Adam or SGD!")

        self.scheduler = CosineAnnealingWarmupRestarts(self.optimG,
                                                       first_cycle_steps=200,  ## cosine cycle 200 epochs
                                                       cycle_mult=1.0,
                                                       max_lr=self.lr * 10.0,
                                                       min_lr=self.lr / 10.0,
                                                       warmup_steps=50,
                                                       ## first 50 epochs in first_cycle_steps as warmup period
                                                       gamma=1.0)  ## scale max_lr

        # for name, p in self.netG.named_parameters():
        #     if p.requires_grad == True:
        #         print (name)

        self.load()

        ## freeze encoder: EfficientNet or Vit ####
        # for name, p in self.netG.named_parameters():
        #     if "encoder" in name:
        #         p.requires_grad = False
        #         print ("freeze ===>%s"%name)
        #     else:
        #         print("trainning ===>%s" % name)
        ## freeze encoder: EfficientNet or Vit ####

        # for name, p in self.netG.named_parameters():
        #     print (name, p.size())

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
                os.path.join(config['train']['save_dir'], 'dis'))
            self.gen_writer = SummaryWriter(
                os.path.join(config['train']['save_dir'], 'gen'))

    ###########################################################################################################################################
    ###################     Distributed_concat() for DDP inference/Evaluation      ############################################################
    ###########################################################################################################################################
    # Allowing Inferences/Evaluation in DDP mode by gathering the results on different GPUs
    # (- DDP in torch only synchronise the gradients for training during bp but not for inference in forward computation;
    #  - TorchMetric can provide a DDP inference/Evaluation but whose results may be different due to the padding data in batches on GPUs;
    #  - This method provide a DDP inference/Evaluation having exactly the same results as the ones obtained by Single GPU by truncating the padding data)
    # 1. Gatherining the data from different GPUS by all_gather，将各个进程中的同一份数据合并到一起。
    #    和all_reduce不同的是，all_reduce是平均，而这里是合并。
    # 2. Truncate the results of the padding data which used to align batch size on different GPUs.
    #    N.B. TorchMetrics does not consider this point, which cause the difference of the inference/evaluation
    #    results on Single GPU mode and DDP mode.
    #    要注意的是，函数的最后会裁剪掉后面额外长度的部分，这是之前的SequentialDistributedSampler添加的。
    # 3. This function requires the output of each GPU has the same size
    #    这个函数要求，输入tensor在各个进程中的大小是一模一样的。
    ############################################################################################################################################
    def distributed_concat(self, tensor, num_total_examples):
        output_tensors = [tensor.clone().cuda() for _ in range(torch.distributed.get_world_size())]
        torch.distributed.all_gather(output_tensors, tensor)
        concat = torch.cat(output_tensors, dim=0)
        # truncate the dummy elements added by SequentialDistributedSampler
        return concat[:num_total_examples]

    def setup_seed(self, seed):
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)
        torch.backends.cudnn.deterministic = True

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
        # new_lr = self.get_lr() * 1.05
        print("Decay learning rate lr ====>%f" % new_lr)
        for param_group in self.optimG.param_groups:
            param_group['lr'] = new_lr
        for param_group in self.optimD.param_groups:
            param_group['lr'] = new_lr

    def reset_learning_rate(self):
        new_lr = self.lr
        print("reset learning rate lr ====> %f" % new_lr)
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
            writer.add_scalar(name, self.summary[name] / 100, self.iteration)
            self.summary[name] = 0

    ## loaded model with current parameters
    def update_model(self, pretrained_model, current_model):
        current_model_dict = current_model.state_dict()
        pretrained_model_select = {k: v for k, v in pretrained_model.items() if
                                   k in current_model_dict and (v.size() == current_model_dict[k].size())}
        for k, v in pretrained_model.items():
            # print("=====k:%s" % (k))
            if k not in current_model_dict:
                print("pretrained model's %s not in current_model!!" % (k))
            else:
                if (v.size() != current_model_dict[k].size()):
                    print("pretrained model's %s size is not equal as the one in current model!!" % (k))

        current_model_dict.update(pretrained_model_select)
        current_model.load_state_dict(current_model_dict)

    # load netG
    def load(self):
        model_path = self.config['train']['save_dir']

        ## load the pretrained model
        if os.path.isfile(self.config['train']['pretrained_model']):
            print("Load pretrained model: %s" % self.config['train']['pretrained_model'])
            gen_path = self.config['train']['pretrained_model']
            pretrained_epoch = str.split(gen_path, '/')[-1][-9:-4]
            dis_path = gen_path[:-13] + "dis_" + pretrained_epoch + ".pth"
            opt_path = gen_path[:-13] + "opt_" + pretrained_epoch + ".pth"
            data = torch.load(gen_path, map_location=self.config['device'])
            self.netG.load_state_dict(data['netG'])
            # data_optimG = torch.load(opt_path, map_location=self.config['device'])
            # self.optimG.load_state_dict(data_optimG['optimG'])
            # self.update_model(data['netG'], self.netG)
            # data = torch.load(opt_path, map_location=self.config['device'])
            # self.update_model(data['optimG'], self.optimG)
            """
            ### Not nessary for inference! But neccessary for fine-tunine!###
            """
            data = torch.load(dis_path, map_location=self.config['device'])
            self.netD.load_state_dict(data['netD'])
            data = torch.load(opt_path, map_location=self.config['device'])
            self.optimG.load_state_dict(data['optimG'])
            self.optimD.load_state_dict(data['optimD'])
            self.epoch = data['epoch']
            if 'epoch_best' in data:
                self.epoch_best = data['epoch_best']
                self.ACER_best = data['ACER_best']
                self.APCER_best = data['APCER_best']
                self.BPCER_best = data['BPCER_best']
                self.ACC_best = data['ACC_best']
            else:
                self.epoch_best = 0
                self.ACER_best = 1.0
                self.APCER_best = 1.0
                self.BPCER_best = 1.0
                self.ACC_best = 0.0
            # self.iteration = data['iteration']
            if self.config['global_rank'] == 0:
                print('Loading pretrained model from {}...'.format(gen_path))
        else:

            if os.path.isfile(os.path.join(model_path, 'latest.ckpt')):
                print("Load pretrained model: %s" % os.path.isfile(os.path.join(model_path, 'latest.ckpt')))
                latest_epoch = open(os.path.join(
                    model_path, 'latest.ckpt'), 'r').read().splitlines()[-1]
                if not os.path.isfile(os.path.join(model_path, 'opt_' + latest_epoch + '.pth')):
                    ckpts = [os.path.basename(i).split('.pth')[0] for i in glob.glob(
                        os.path.join(model_path, '*.pth'))]
                    ckpts.sort()
                    latest_epoch = str.split(ckpts[-1], '_')[-1] if len(ckpts) > 0 else None
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
                print("Load pretrained model: %s" % gen_path)
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
                self.epoch_best = data['epoch_best']
                self.ACER_best = data['ACER_best']
                self.APCER_best = data['APCER_best']
                self.BPCER_best = data['BPCER_best']
                self.ACC_best = data['ACC_best']
                self.iteration = data['iteration']
            else:
                if self.config['global_rank'] == 0:
                    print(
                        'Warnning: There is no trained model found. An initialized model will be used.')

    # save parameters every eval_epoch
    def save(self, epoch):
        if self.config['global_rank'] == 0:
            gen_path = os.path.join(
                self.config['train']['save_dir'], 'gen_{}.pth'.format(str(epoch).zfill(5)))
            dis_path = os.path.join(
                self.config['train']['save_dir'], 'dis_{}.pth'.format(str(epoch).zfill(5)))
            opt_path = os.path.join(
                self.config['train']['save_dir'], 'opt_{}.pth'.format(str(epoch).zfill(5)))
            print('\nsaving model to {} ...'.format(gen_path))
            if isinstance(self.netG, torch.nn.DataParallel) or isinstance(self.netG, DDP):
                netG = self.netG.module
                netD = self.netD.module
            else:
                netG = self.netG
                netD = self.netD

            saved_gen = glob.glob(os.path.join(self.config['train']['save_dir'], 'gen_*.pth'))
            saved_gen.sort()
            saved_dis = glob.glob(os.path.join(self.config['train']['save_dir'], 'dis_*.pth'))
            saved_dis.sort()
            saved_opt = glob.glob(os.path.join(self.config['train']['save_dir'], 'opt_*.pth'))
            saved_opt.sort()

            torch.save({'netG': netG.state_dict()}, gen_path)
            torch.save({'netD': netD.state_dict()}, dis_path)
            torch.save({'epoch': self.epoch,
                        'epoch_best': self.epoch_best,
                        'ACER_best': self.ACER_best,
                        'APCER_best': self.APCER_best,
                        'BPCER_best': self.BPCER_best,
                        'ACC_best': self.ACC_best,
                        'iteration': self.iteration,
                        'optimG': self.optimG.state_dict(),
                        'optimD': self.optimD.state_dict()}, opt_path)
            os.system('echo {} > {}'.format(str(epoch).zfill(5),
                                            os.path.join(self.config['train']['save_dir'], 'latest.ckpt')))

            if len(saved_gen) > 0:
                for i in range(len(saved_gen)):
                    if int(saved_gen[i][-9:-4]) != self.epoch_best:
                        ## delete the old gen_*.pth, opt_*.pth, dis_*.pth only save the latest dis.pth and opt.pth
                        os.system('rm {}'.format(saved_gen[i]))
                        os.system('rm {}'.format(saved_dis[i]))
                        os.system('rm {}'.format(saved_opt[i]))

    def grad_cam(self, batch_size, input_tensor, input_mask, target_category, rgb_img):

        # rgb_img=cv2.imread(rgb_img, 1)[:, :, ::-1]
        # rgb_img=cv2.resize(rgb_img, (224, 224))
        # rgb_img=np.float32(rgb_img) / 255

        self.cam.batch_size = batch_size
        grayscale_cam = self.cam(input_tensor=input_tensor,
                                 input_mask=input_mask,
                                 target_category=target_category,
                                 eigen_smooth=False,
                                 aug_smooth=False)

        # Here grayscale_cam has only one image in the batch
        grayscale_cam = grayscale_cam[0, :]

        cam_image = show_cam_on_image(rgb_img, grayscale_cam)
        cv2.imwrite(f'{self.grad_cam_method}_cam.jpg', cam_image)

    def get_fast_weights(self, grad, params):
        fast_weights = []
        for p in zip(grad, params):
            if p[0] is not None:
                fast_weights.append(p[1] - self.lr * p[0])
            else:
                fast_weights.append(p[1])
        return  fast_weights

    def meta_inner_update(self, fast_weights):
        assert len(fast_weights) ==  len([p for p in self.netG.parameters()])
        with torch.no_grad():
            for i, p in enumerate(self.netG.parameters()):
                assert len(p) == len(fast_weights[i])
                p.copy_(fast_weights[i])

    # train entry
    def train(self):
        pbar = range(int(self.train_args['iterations']))
        # if self.config['global_rank'] == 0:
        #     pbar = tqdm(pbar, initial=self.iteration, dynamic_ncols=True, smoothing=0.01)

        with open(os.path.join(self.save_dir, "log_train.txt"), "w") as train_log_file:
            numEpoch_lr_reduce = 2  ### ACER as indicator: 2 i.e. 2x5=10 epoch ### loss_train as indicator: continous 2 epochs loss increase
            numEpoch_lr_reset = 200
            lr_indicator0 = 0  ### ACER as indicator 0 ### loss_train as indicator 1000

            nepoch_increse = 0

            while True:
                self.epoch += 1
                if self.config['distributed']:
                    self.meta_train_sampler.set_epoch(self.epoch)

                loss_train_avg = self._train_epoch(pbar, train_log_file)
                if self.iteration > self.train_args['iterations']:
                    break

                # saving models and test model
                if self.epoch % 1 == 0:  ## debug or test
                # if self.epoch % 1 == 0: ## image-wise
                # if self.epoch % 5 == 0: ## video-wise
                    ACC, APCER, BPCER, ACER = self.test()
                    self.save(int(self.epoch))



    # process input and calculate loss every training epoch
    def _train_epoch(self, pbar, train_log_file):

        tt = [0.0, 0.0]
        device = self.config['device']
        self.netG.train()
        self.netD.train()

        ## learning rate decay in 200 epochs period: warm-up (50 epochs)+cosine lr decay (50-150 epochs)
        self.scheduler.step()

        Loss_train = []

        ## each epoch has 1000 iterations ##
        for j in range(2):
            loss_meta_test_alltask = [torch.tensor(0.0), torch.tensor(0.0), torch.tensor(0.0)]
            loss_meta_test_alltask0 = 0
            loss_meta_train0 = 0
            #### the initial model of each iteration ####
            #netG_0 = deepcopy(self.netG)

            ##############################################################################################
            ## each iteration of meta_learning has to train on N tasks  for K steps (self.update_step)  ##
            ## each configuration of meta_train/meta_test datasets is a task                            ##
            ##############################################################################################
            ### len(meta_train_loader_list) is the task number ###
            task_num = len(self.meta_train_loader_list)
            for i in range(task_num):
                meta_train_loader = self.meta_train_loader_list[i]
                meta_test_loader = self.meta_test_loader_list[i]

                ####################################################################
                ##  create a _DataLoaderIter outside the loop and iterate over it ##
                ##  do not use batch = next(iter(meta_train_loader)) in the loop, ##
                ##  since this will create an object of class _DataLoaderIter n times and retrieve the first batch only ##
                ####################################################################
                dataloader_iterator = iter(meta_train_loader)

                ### self.update_step (e.g. 10) is the steps training on the support set (meta_train dataset) of taks i  ####
                # 1. run the i-th task and compute loss for k=0
                batch_idx = 0
                try:
                    batch = next(dataloader_iterator)
                except StopIteration:
                    ## once all batches are retrieved, _DataLoaderIter will raise a StopIteration error ##
                    ## so once all batches are retrieved, create again a _DataLoaderIter                ##
                    dataloader_iterator = iter(meta_train_loader)
                    batch = next(dataloader_iterator)
                loss_meta_train, Loss_train = self.forward(batch, device, Loss_train)
                grad = torch.autograd.grad(loss_meta_train, self.netG.parameters(), allow_unused=True)
                fast_weights = self.get_fast_weights(grad, self.netG.parameters())
                #fast_weights = list(map(lambda p: p[1] - self.lr * p[0], zip(grad, self.netG.parameters())))
                self.meta_inner_update(fast_weights)

                for k in range(1, self.update_step): ## update_step 10
                    batch_idx = k
                    try:
                        batch = next(dataloader_iterator)
                    except StopIteration:
                        ## once all batches are retrieved, _DataLoaderIter will raise a StopIteration error ##
                        ## so once all batches are retrieved, create again a _DataLoaderIter                ##
                        dataloader_iterator = iter(meta_train_loader)
                        batch = next(dataloader_iterator)

                    ##self.adjust_learning_rate() ## lr searching test
                    # ##debug
                    # if batch_idx > 3:
                    #     break
                    # self.adjust_learning_rate()
                    self.iteration += 1

                    loss_meta_train, Loss_train = self.forward(batch, device, Loss_train)

                    ### inner update of meta learning ###
                    # self.optimG.zero_grad()
                    # loss_meta_train.backward()
                    # self.optimG.step()
                    grad = torch.autograd.grad(loss_meta_train, fast_weights, allow_unused=True)
                    #fast_weights = list(map(lambda p: p[1] - self.lr*p[0], zip(grad, fast_weights)))
                    fast_weights = self.get_fast_weights(grad, fast_weights)
                    ## update model with fast_weights
                    self.meta_inner_update(fast_weights)

                    ### print results ###
                    self.print_results(batch_idx, tt, 1000*task_num*self.update_step, train_log_file, loss_meta_train, batch)

                ############################################
                ### calculate the loss on the query set ####
                ############################################
                dataloader_iterator = iter(meta_test_loader)
                batch_idx = k+1
                try:
                    batch = next(dataloader_iterator)
                except StopIteration:
                    ## once all batches are retrieved, _DataLoaderIter will raise a StopIteration error ##
                    ## so once all batches are retrieved, create again a _DataLoaderIter                ##
                    dataloader_iterator = iter(meta_train_loader)
                    batch = next(dataloader_iterator)

                loss_meta_test, Loss_train = self.forward(batch, device, Loss_train)
                loss_meta_test_alltask[i] = loss_meta_test
                loss_meta_test_alltask0 += loss_meta_test.item()

                ### print results ###
                self.print_results(batch_idx, tt, 1000*task_num*self.update_step, train_log_file, loss_meta_test, batch)

                ###################################################################
                ### Make the model return to the intial state for the next task ###
                ###################################################################
                # del self.netG
                # self.netG = deepcopy(netG_0)

            ###############################################################################################
            ### update model by the mean value of query losses on all tasks for iteration j.           ####
            ### this is the real update of the model , which is also called outter meta optimisation   ####
            ###############################################################################################

            loss_q =  sum(loss_meta_test_alltask) / task_num
            #loss_q =  (loss_meta_test_alltask[1]+loss_meta_test_alltask[2]) / 2
            self.optimG.zero_grad()
            loss_q.backward()
            self.optimG.step()

        return sum(Loss_train) / len(Loss_train)


    def forward(self, batch, device, Loss_train):
        # frames, masks, spoofing_labels = batch['frame_tensors'].to(device), batch['mask_tensors'].to(device),  batch['spoofing_labels']
        frames, masks, spoofing_labels = batch['frame_tensors'].to(device, non_blocking=True), batch[
            'mask_tensors'].to(device, non_blocking=True), batch['spoofing_labels']

        b, t, c, h, w = frames.size()
        masked_frame = (frames * (1 - masks).float())
        y_enc = self.netG(masked_frame, masks)

        # ############ calcluate GFLOPS and Parameters #############
        # flops, params=profile(self.netG, inputs=(masked_frame, masks))
        # print("GFlops: %f Params %f"%(flops/1e9, params/1e6))
        # ############################

        ### transformer binary classification entropy loss (Transformer BCE)
        images_labels = np.array([t * [label] for label in spoofing_labels])
        images_labels = torch.from_numpy(images_labels.flatten())
        images_labels = images_labels.to(device)
        loss_softmax_enc = self.bce_loss(y_enc, images_labels)
        Loss_train.append(loss_softmax_enc)
        # loss_softmax_depth = self.bce_loss(y_dec, images_labels)
        self.add_summary(
            self.gen_writer, 'loss/loss_softmax_enc', loss_softmax_enc.item()
        )
        # self.add_summary(
        #     self.gen_writer, 'loss/loss_softmax_depth', loss_softmax_depth.item()
        # )

        # bp() ## debug
        gen_loss = 0
        dis_loss = 0

        # gen_loss += 1.0*loss_softmax_enc + 1.0*loss_softmax_depth
        gen_loss += 1.0 * loss_softmax_enc

        return gen_loss, Loss_train

    def print_results(self, batch_idx, tt, epoch_size, train_log_file, gen_loss, batch):

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

        if self.config['global_rank'] == 0:
            tt[self.iteration % 2] = time.time()
            print(
                'Train Epoch: {}/{} [ {}/{} batch)] time:{:.2f} \ttransformer_softmax_loss: {:.6f} lr: {:.6f}'.format(
                    int(self.epoch), int(self.train_args['iterations'] / epoch_size), (batch_idx),
                    epoch_size,
                    abs(tt[-1] - tt[-2]), gen_loss.item(), self.get_lr()))
            print("spoofing labels of videos in batch:", end="")
            for spoofing_label in batch['spoofing_labels']:
                print("%d " % spoofing_label.item(), end="")
            print("length batch['spoofing_labels'] : %d" % len(batch['spoofing_labels']))
            print("ACC_best=%.4f, APCER_best=%.4f, BPCER_best=%.4f, ACER_best=%.4f  epoch_best=%d\n" \
                  % (self.ACC_best, self.APCER_best, self.BPCER_best, self.ACER_best, self.epoch_best))
            print('\n')

            ## save training info in log_train.txt
            train_log_file.write(
                'Train Epoch: {}/{} [ {}/{} batch)] time:{:.2f} transformer_softmax_loss: {:.6f} lr: {:.6f}\n'.format(
                    int(self.epoch), int(self.train_args['iterations'] / epoch_size), (batch_idx),
                    epoch_size,
                    abs(tt[-1] - tt[-2]), gen_loss.item(), self.get_lr()))
            spoofing_labels = []
            for spoofing_label in batch['spoofing_labels']:
                spoofing_labels.append(spoofing_label.item())
            train_log_file.write("spoofing labels of videos in batch: %s\n" % spoofing_labels)
            train_log_file.write("length batch['spoofing_labels'] : %d\n" % len(batch['spoofing_labels']))
            train_log_file.write("ACC_best=%.4f, APCER_best=%.4f, BPCER_best=%.4f, ACER_best=%.4f  epoch_best=%d\n" \
                                 % (
                                     self.ACC_best, self.APCER_best, self.BPCER_best, self.ACER_best, self.epoch_best))
            train_log_file.write('\n')



    def test(self):
        # pbar = range(int(self.train_args['iterations']))
        # if self.config['global_rank'] == 0:
        #     pbar = tqdm(pbar, initial=self.iteration, dynamic_ncols=True, smoothing=0.01)

        # self.epoch += 1
        # if self.config['distributed']:
        #     self.test_sampler.set_epoch(self.epoch)

        with open(os.path.join(self.save_dir, "evaluation_ep%d.txt" % self.epoch), "w") as f:

            y_labels, gt_labels = self._test_epoch()

            ACC, APCER, BPCER, ACER = self.performance(y_labels, gt_labels)
            print("ACC=%.4f, APCER=%.4f, BPCER=%.4f, ACER=%.4f  \n" % (ACC, APCER, BPCER, ACER))
            if ACER < self.ACER_best:
                self.ACC_best = ACC
                self.APCER_best = APCER
                self.BPCER_best = BPCER
                self.ACER_best = ACER
                self.epoch_best = self.epoch
            if self.config['global_rank'] == 0:
                print("\n")
                #print("ACC=%.4f, APCER=%.4f, BPCER=%.4f, ACER=%.4f  \n" % (ACC, APCER, BPCER, ACER))
                f.write("ACC=%.4f, APCER=%.4f, BPCER=%.4f, ACER=%.4f  \n" % (ACC, APCER, BPCER, ACER))
                f.write("ACC_best=%.4f, APCER_best=%.4f, BPCER_best=%.4f, ACER_best=%.4f, epoch_best=%d  \n" % (
                self.ACC_best, self.APCER_best, self.BPCER_best, self.ACER_best, self.epoch_best))

        return ACC, APCER, BPCER, ACER

    def performance(self, y_labels, labels, threshold=0.5):
        if self.classes == 2:
            idx_type1 = [i for i, y in enumerate(y_labels) if
                         y < threshold and labels[i] == 1]  ## false negative/attack, Bonafide label 1
            idx_type2 = [i for i, y in enumerate(y_labels) if
                         y > threshold and labels[i] == 0]  ## false positive/Bonafide, attack label 0
        elif self.classes > 2:
            ## WMCA unseen protocol, Bonafide 0, Attack type 1-4
            idx_type1 = [i for i, y in enumerate(y_labels) if
                         y != 0 and labels[i] == 0]  ## false negative/attack: 1-4 attack type
            idx_type2 = [i for i, y in enumerate(y_labels) if
                         y == 0 and labels[i] != 0]  ## false positive/Bonafide, 0 Bonafide label
        else:
            raise ValueError("Classes number %d is not correct!" % self.classes)

        type1 = len(idx_type1)  ## False negative (False attack)
        type2 = len(idx_type2)  ## False positive (False live)

        count = len(labels)
        if self.classes == 2:
            num_real = sum(labels)
            num_fake = count - num_real
            ACC = 1 - (type1 + type2) / count
        elif self.classes > 2:
            num_real = len([label for label in labels if label == 0])
            num_fake = count - num_real
            ACC = 1 - (type1 + type2) / count
        else:
            raise ValueError("Classes number %d is not correct!" % self.classes)

        if num_fake == 0:
            APCER = 0
        else:
            APCER = type2 / num_fake  ## false liveness rate/false acceptance rate

        if num_real == 0:
            BPCER = 0
        else:
            BPCER = type1 / num_real  ## false attack rate/ false rejection rate

        ACER = (APCER + BPCER) / 2.0

        return ACC, APCER, BPCER, ACER

    # process input and calculate loss every training epoch
    def _test_epoch(self):
        tt = [0.0, 0.0]
        device = self.config['device']
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
            for batch_idx, batch in enumerate(self.test_loader):

                self.netG.eval()
                self.iteration += 1
                print(batch_idx)
                frames, masks, spoofing_labels, images_path = batch['frame_tensors'].to(device), batch[
                    'mask_tensors'].to(device), batch['spoofing_labels'], batch['images_path']
                # frames, masks, spoofing_labels = batch['frame_tensors'].to(device), batch['mask_tensors'].to(device),  batch['spoofing_labels']

                y_labels, images_labels_flat = self.infer(frames, masks, spoofing_labels)
                # y_labels, images_labels_flat = self.infer_neighbour(frames, masks, spoofing_labels)

                ## acc = self.metric(torch.as_tensor(y_labels), torch.as_tensor(images_labels_flat))
                ## #if self.config['global_rank'] == 0:  # print only for rank 0
                ## print(f"Accuracy on batch {batch_idx}: {acc} gpu rank: {self.config['global_rank']}")

                Y_labels += y_labels
                GT_labels += images_labels_flat

                if self.config['global_rank'] == 0:
                    print(">>>>>> Evaluating epoch %d...%d/%d batches, batch_size %d" % (
                    self.epoch, batch_idx, len(self.test_loader), len(spoofing_labels)), end="\r")

            print("\n")
            ## Gathering the results of the different GPUs
            Y_labels = self.distributed_concat(torch.as_tensor(Y_labels).cuda(),len(self.test_dataset)*self.config['data_loader']['sample_length'])
            GT_labels = self.distributed_concat(torch.as_tensor(GT_labels).cuda(),len(self.test_dataset)*self.config['data_loader']['sample_length'])

            # # metric on all batches and all gpus using custom accumulation
            # # accuracy is same across all gpus
            # acc = self.metric.compute()
            # print(f"Accuracy on all data: {acc}, gpu rank: {self.config['global_rank']}")

            # ## reset for next epoch
            # self.metric.reset()

        return Y_labels, GT_labels

    def infer(self, frames, masks, spoofing_labels):
        sample_length = self.sample_length
        b, t, c, h, w = frames.size()
        if t > sample_length:
            start_index_list = list(range(t)[::sample_length])
            select_groups = 4

            for i in start_index_list[:select_groups]:
                if (i + sample_length) <= t:
                    frames_ = frames[:, i:i + sample_length, :, :, :]
                    masks_ = masks[:, i:i + sample_length, :, :, :]

                    masked_frame = (frames_ * (1 - masks_).float())
                    y_enc = self.netG(masked_frame, masks_)
                    y_labels = [np.argmax(y) for y in y_enc.detach().cpu().tolist()]
                    images_labels = np.array([sample_length * [label] for label in spoofing_labels])
                    images_labels_flat = [item for sublist in images_labels for item in sublist]
        else:
            masked_frame = (frames * (1 - masks).float())
            y_enc = self.netG(masked_frame, masks)
            y_labels = [np.argmax(y) for y in y_enc.detach().cpu().tolist()]
            images_labels = np.array([t * [label] for label in spoofing_labels])
            images_labels_flat = [item for sublist in images_labels for item in sublist]

        return y_labels, images_labels_flat

    def infer_neighbour(self, frames, masks, spoofing_labels):
        sample_length = self.sample_length
        b, t, c, h, w = frames.size()
        video_length = t
        Y_enc = [None] * video_length
        with torch.no_grad():
            for f in range(0, video_length, sample_length):
                neighbor_ids = [i for i in range(max(0, f - sample_length), min(video_length, f + sample_length + 1))]
                ## calculate the encoder feats of each video clip which do not require large gpu memoery but with low efficience
                # infer_frames_ids = neighbor_ids + ref_ids
                frames_ = frames[:, neighbor_ids, :, :, :]
                masks_ = masks[:, neighbor_ids, :, :, :]

                masked_frame = (frames_ * (1 - masks_).float())
                y_enc = self.netG(masked_frame, masks_)

                for i in range(len(neighbor_ids)):
                    idx = neighbor_ids[i]
                    if Y_enc[idx] is None:
                        Y_enc[idx] = y_enc[i]
                    else:
                        Y_enc[idx] = Y_enc[idx] * 0.5 + y_enc[i] * 0.5

            y_labels = [np.argmax(y.detach().cpu().tolist()) for y in Y_enc]
            images_labels = np.array([video_length * [label] for label in spoofing_labels])
            images_labels_flat = [item for sublist in images_labels for item in sublist]

        return y_labels, images_labels_flat