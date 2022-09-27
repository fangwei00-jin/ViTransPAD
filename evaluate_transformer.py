#from core.Evaluate import Evaluate
#from core.Evaluate_transformer_EfficientNet_global import Evaluate
#from core.Evaluate_transformer_EfficientNet_local import Evaluate
#from core.Evaluate_transformer_EfficientNet_localglobal import Evaluate
#from core.Evaluate_transformer import Evaluate
#from core.Evaluate_transformer_video import Evaluate
#from core.Evaluate_transformer_EfficientNet_local_vid import Evaluate

import importlib
import argparse
import os
import json
from shutil import copyfile
import matplotlib
matplotlib.use('Agg')
#matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description='STTN')
parser.add_argument('-c', '--config', default='configs/SiW.json', type=str)

args = parser.parse_args()




def main_worker(config):
    Evaluate = importlib.import_module(config['test']['eval_module'])

    config_path = os.path.join(
        config['save_dir'], config['config'].split('/')[-1])
    if not os.path.isfile(config_path):
        copyfile(config['config'], config_path)

    os.environ['CUDA_VISIBLE_DEVICES'] = config['test']['gpu']

    dataset = os.path.basename(args.config).split('.')[0]
    config['dataset'] = dataset

    ACC_all = []
    APCER_all= []
    BPCER_all = []
    ACER_all = []
    save_dir =config['save_dir']
    model_dir = config['test']['ckpt']
    [first_model, last_model]= config['test']['test_model']
    evaluate = Evaluate.Evaluate(config, 'test')

    if first_model == last_model:
        last_model = first_model +1
    for i in range(first_model,last_model+1, config['test']['test_model_sample_interval']):
        trained_model = model_dir+"gen_%05d"%i+".pth"
        ep = i
        config['test']['ckpt'] = trained_model
        # save_dir_i = os.path.join(save_dir, "%04d"%i)
        # if not os.path.isdir(save_dir_i):
        #     os.mkdir(save_dir_i)
        #     config['save_dir'] = save_dir_i

        ACC, APCER, BPCER, ACER = evaluate.evaluate(ep, config['test']['ckpt'])
        ACC_all.append(ACC)
        APCER_all.append(APCER)
        BPCER_all.append(BPCER)
        ACER_all.append(ACER)


        print('ACC= %.4f, APCER= %.4f, BPCER= %.4f, ACER= %.4f' % (
                 ACC, APCER, BPCER, ACER))
        with open(os.path.join(config['save_dir'], 'ep_%03d'%ep, 'log.txt'), 'w') as f:
            f.write('ACC= %.4f, APCER= %.4f, BPCER= %.4f, ACER= %.4f\n' % (
             ACC, APCER, BPCER, ACER))

    with open(os.path.join(save_dir, 'ACC_all.txt'), 'w') as f:
        f.write('ACC_all\n')
        for i, acc in enumerate(ACC_all):
            f.write('%f\n' % (acc))
    with open(os.path.join(save_dir, 'APCER_all.txt'), 'w') as f:
        f.write('APCER_all\n')
        for i, apcer in enumerate(APCER_all):
            f.write('%f\n' % (apcer))
    with open(os.path.join(save_dir, 'BPCER_all.txt'), 'w') as f:
        f.write('BPCER_all\n')
        for i, bpcer in enumerate(BPCER_all):
            f.write('%f\n' % (bpcer))
    with open(os.path.join(save_dir, 'ACER_all.txt'), 'w') as f:
        f.write('ACER_all\n')
        for i, acer in enumerate(ACER_all):
            f.write('%f\n' % (acer))


    return

if __name__ == '__main__':

    config = json.load(open(args.config))
    config['config'] = args.config


    main_worker(config)
