import os
import json
import argparse
import datetime
import numpy as np
from shutil import copyfile
import torch
import torch.multiprocessing as mp
import importlib

#from core.trainer_gan import Trainer
#from core.trainer_gan_MTL import Trainer
#from core.trainer_gan_MTL_full import Trainer
#from core.trainer_gan_ResNeXt50 import Trainer
#from core.trainer_transformer import Trainer
#from core.trainer_transformer_vid_cls import Trainer
#from core.trainer_mtl import Trainer
#from core.trainer_transformer import Trainer
#from core.trainer_generator import Trainer
#from core.trainer_generator_MTL import Trainer
#from core.trainer_generator_MTL_full import Trainer
#from core.trainer_generator_MTL_full_dynamic import Trainer

from core.dist import (
    get_world_size,
    get_local_rank,
    get_global_rank,
    get_master_ip,
)


debug = 0



#os.environ['CUDA_VISIBLE_DEVICES'] = '3' ## debug






parser = argparse.ArgumentParser(description='STTN')
parser.add_argument('-c', '--config', default='configs/SiW.json', type=str)
parser.add_argument('-m', '--model', default='sttn', type=str)
parser.add_argument('-p', '--port', default='23455', type=str)
parser.add_argument('-e', '--exam', action='store_true')
parser.add_argument('-g', '--global_rank', default=0, type=int)
parser.add_argument('-l', '--local_rank', default=0, type=int)
parser.add_argument('-w', '--world_size', default=1, type=int)
parser.add_argument('-i', '--init_method', default='tcp://localhost:12345', type=str)
args = parser.parse_args()


def main_worker(rank, config):
    Test = importlib.import_module(config['test']['test_module'])

    if 'local_rank' not in config:
        config['local_rank'] = config['global_rank'] = rank
    if config['distributed']:
        print('............................%s' % config['local_rank'])
        torch.cuda.set_device(int(config['local_rank']))

        torch.distributed.init_process_group(backend='nccl', ## gloo
                                             init_method=config['init_method'],
                                             world_size=config['world_size'],
                                             rank=config['global_rank'],
                                             group_name='mtorch'
                                             )
        # print('............................%s'%config['init_method'])
        # torch.distributed.init_process_group(backend='nccl',
        #                                      init_method=config['init_method']
        #                                      )
        print('using GPU {}-{} for training'.format(
            int(config['global_rank']), int(config['local_rank'])))

    config['test']['save_dir'] = os.path.join(config['test']['save_dir'], '{}_{}'.format(config['data_loader']['model'],
                                                                         os.path.basename(args.config).split('.')[0]))
    if torch.cuda.is_available():
        config['device'] = torch.device("cuda:{}".format(config['local_rank']))
    else:
        config['device'] = 'cpu'

    if (not config['distributed']) or config['global_rank'] == 0:
        os.makedirs(config['test']['save_dir'], exist_ok=True)
        config_path = os.path.join(
            config['test']['save_dir'], config['config'].split('/')[-1])
        # if not os.path.isfile(config_path):
        if config['config'] != config_path:
            copyfile(config['config'], config_path)
        print('[**] create folder {}'.format(config['test']['save_dir']))

    test = Test.Test(config, debug=args.exam)
    test.test()




if __name__ == "__main__":
    
    # loading configs
    config = json.load(open(args.config))
    # config['model'] = args.model
    config['config'] = args.config

    # setting distributed configurations
    # config['world_size'] = get_world_size()
    config['world_size'] = args.world_size
    #config['init_method'] = f"tcp://{get_master_ip()}:{args.port}"
    config['init_method'] = args.init_method
    if debug:
        config['distributed'] = False
    else:
        config['distributed'] = True if config['world_size'] > 0 else False


    # setup distributed parallel training environments
    #if get_master_ip() == "127.0.0.1":
    if "127.0.0.1" in args.init_method:
        # manually launch distributed processes
        print(".................127")
        mp.spawn(main_worker, nprocs=config['world_size'], args=(config,))
    else:
        # multiple processes have been launched by openmpi 
        # config['local_rank'] = get_local_rank()
        # config['global_rank'] = get_global_rank()
        config['global_rank'] = args.global_rank
        config['local_rank'] = args.local_rank
        main_worker(-1, config)
