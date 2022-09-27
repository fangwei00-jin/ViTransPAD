#from core.Evaluate import Evaluate
#from core.Evaluate_transformer import Evaluate
#from core.Evaluate_singletask_EfficientNet import Evaluate
from core.Evaluate_singletask_EfficientNet_vid import Evaluate

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
    threshold_all=[]
    save_dir =config['save_dir']
    model_dir = config['test']['ckpt']
    [first_model, last_model]= config['test']['test_model']
    evaluate = Evaluate(config, 'test')

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

        # ACC, APCER, BPCER, ACER, threshold, fprs, tprs, thresholds, acc_list, thresholds_test = evaluate.evaluate(ep, config['test']['ckpt'])
        # ACC_all.append(ACC)
        # APCER_all.append(APCER)
        # BPCER_all.append(BPCER)
        # ACER_all.append(ACER)
        # threshold_all.append(threshold)
        #
        #
        # print('threshold= %.4f, ACC= %.4f, APCER= %.4f, BPCER= %.4f, ACER= %.4f' % (
        #         threshold, ACC, APCER, BPCER, ACER))
        # with open(os.path.join(config['save_dir'], 'ep_%03d'%ep, 'log.txt'), 'w') as f:
        #     f.write('threshold= %.4f, ACC= %.4f, APCER= %.4f, BPCER= %.4f, ACER= %.4f\n' % (
        #     threshold, ACC, APCER, BPCER, ACER))
        # with open(os.path.join(config['save_dir'], 'ep_%03d'%ep, 'val_fpr.txt'), 'w') as f:
        #     f.write('fps\n')
        #     for i, fpr in enumerate(fprs):
        #         f.write('%f\n' %(fpr))
        # with open(os.path.join(config['save_dir'], 'ep_%03d'%ep, 'val_tpr.txt'), 'w') as f:
        #     f.write('tprs\n')
        #     for i, fpr in enumerate(fprs):
        #         f.write('%f\n' %(tprs[i]))
        # with open(os.path.join(config['save_dir'], 'ep_%03d'%ep, 'val_threholds.txt'), 'w') as f:
        #     f.write('thresholds\n')
        #     for i, fpr in enumerate(fprs):
        #         f.write('%f\n' %(thresholds[i]))
        # with open(os.path.join(config['save_dir'], 'ep_%03d'%ep, 'acc_test.txt'), 'w') as f:
        #     f.write('acc\n')
        #     for i, acc in enumerate(acc_list):
        #         f.write('%f\n' %(acc))
        # with open(os.path.join(config['save_dir'], 'ep_%03d'%ep, 'threholds_test.txt'), 'w') as f:
        #     f.write('thresholds_test\n')
        #     for i, acc in enumerate(acc_list):
        #         f.write('%f\n' %(thresholds_test[i]))

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
    with open(os.path.join(save_dir, 'threshold_all.txt'), 'w') as f:
        f.write('threshold_all\n')
        for i, threshold in enumerate(threshold_all):
            f.write('%f\n' % (threshold))

    return

if __name__ == '__main__':

    config = json.load(open(args.config))
    config['config'] = args.config


    main_worker(config)
