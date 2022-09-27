#from core.Evaluate import Evaluate
# from core.Evaluate_transformer import Evaluate
#from core.Evaluate_MTL_EfficientNet import Evaluate
from core.Evaluate_MTL_EfficientNet_full import Evaluate


import argparse
import os
import json
from shutil import copyfile
import matplotlib
#matplotlib.use('Agg')
#import matplotlib.pyplot as plt

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

    ACC_all_val = []
    APCER_all_val= []
    BPCER_all_val = []
    ACER_all_val = []


    ACC_enc_all_val = []
    APCER_enc_all_val= []
    BPCER_enc_all_val = []
    ACER_enc_all_val = []


    ACC_dec_all_val = []
    APCER_dec_all_val = []
    BPCER_dec_all_val = []
    ACER_dec_all_val = []


    ACC_all = []
    APCER_all= []
    BPCER_all = []
    ACER_all = []
    threshold_all=[]

    ACC_enc_all = []
    APCER_enc_all= []
    BPCER_enc_all = []
    ACER_enc_all = []
    threshold_all_enc_cls = []

    ACC_dec_all = []
    APCER_dec_all= []
    BPCER_dec_all = []
    ACER_dec_all = []
    threshold_all_depth_cls = []

    ACC_vote_all = []
    APCER_vote_all= []
    BPCER_vote_all = []
    ACER_vote_all = []

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

        # ACC, APCER, BPCER, ACER, threshold, fprs, tprs, thresholds, acc_list, thresholds_test, \
        # ACC_cls_enc, APCER_cls_enc, BPCER_cls_enc, ACER_cls_enc, \
        # ACC_cls_dec, APCER_cls_dec, BPCER_cls_dec, ACER_cls_dec,\
        # ACC_vote, APCER_vote, BPCER_vote, ACER_vote= evaluate.evaluate(ep, config['test']['ckpt'])

        threshold, ACC_val, APCER_val, BPCER_val, ACER_val, APCERs_val, BPCERs_val, \
        threshold_enc_cls, ACC_val_enc_cls, APCER_val_enc_cls, BPCER_val_enc_cls, ACER_val_enc_cls, APCERs_val_enc_cls, BPCERs_val_enc_cls, \
        threshold_depth_cls, ACC_val_depth_cls, APCER_val_depth_cls, BPCER_val_depth_cls, ACER_val_depth_cls, APCERs_val_depth_cls, BPCERs_val_depth_cls, \
        ACC, APCER, BPCER, ACER, \
        ACC_cls_enc, APCER_cls_enc, BPCER_cls_enc, ACER_cls_enc, \
        ACC_cls_dec, APCER_cls_dec, BPCER_cls_dec, ACER_cls_dec, \
        ACC_vote, APCER_vote, BPCER_vote, ACER_vote = evaluate.evaluate(ep, config['test']['ckpt'])

        ACC_all_val.append(ACC_val)
        APCER_all_val.append(APCER_val)
        BPCER_all_val.append(BPCER_val)
        ACER_all_val.append(ACER_val)


        ACC_enc_all_val.append(ACC_val_enc_cls)
        APCER_enc_all_val.append(APCER_val_enc_cls)
        BPCER_enc_all_val.append(BPCER_val_enc_cls)
        ACER_enc_all_val.append(ACER_val_enc_cls)

        ACC_dec_all_val.append(ACC_val_depth_cls)
        APCER_dec_all_val.append(APCER_val_depth_cls)
        BPCER_dec_all_val.append(BPCER_val_depth_cls)
        ACER_dec_all_val.append(ACER_val_depth_cls)


        ACC_all.append(ACC)
        APCER_all.append(APCER)
        BPCER_all.append(BPCER)
        ACER_all.append(ACER)
        threshold_all.append(threshold)

        ACC_enc_all.append(ACC_cls_enc)
        APCER_enc_all.append(APCER_cls_enc)
        BPCER_enc_all.append(BPCER_cls_enc)
        ACER_enc_all.append(ACER_cls_enc)
        threshold_all_enc_cls.append(threshold_enc_cls)

        ACC_dec_all.append(ACC_cls_dec)
        APCER_dec_all.append(APCER_cls_dec)
        BPCER_dec_all.append(BPCER_cls_dec)
        ACER_dec_all.append(ACER_cls_dec)
        threshold_all_depth_cls.append(threshold_depth_cls)

        ACC_vote_all.append(ACC_vote)
        APCER_vote_all.append(APCER_vote)
        BPCER_vote_all.append(BPCER_vote)
        ACER_vote_all.append(ACER_vote)

        print("""threshold= %.4f, ACC= %.4f, APCER= %.4f, BPCER= %.4f, ACER= %.4f\n
              ACC_enc= %.4f, APCER_enc= %.4f, BPCER_enc= %.4f, ACER_enc= %.4f\n
              ACC_dec= %.4f, APCER_dec= %.4f, BPCER_dec= %.4f, ACER_dec= %.4f\n
              ACC_vote= %.4f, APCER_vote= %.4f, BPCER_vote= %.4f, ACER_vote= %.4f\n""" % (
                threshold, ACC, APCER, BPCER, ACER, \
                ACC_cls_enc, APCER_cls_enc, BPCER_cls_enc, ACER_cls_enc, \
                ACC_cls_dec, APCER_cls_dec, BPCER_cls_dec, ACER_cls_dec, \
                ACC_vote, APCER_vote, BPCER_vote, ACER_vote))
        with open(os.path.join(config['save_dir'], 'ep_%03d'%ep, 'log.txt'), 'w') as f:
            f.write("""threshold= %.4f, ACC= %.4f, APCER= %.4f, BPCER= %.4f, ACER= %.4f\n 
              threshold_enc_cls= %.4f, ACC_enc= %.4f, APCER_enc= %.4f, BPCER_enc= %.4f, ACER_enc= %.4f\n 
              threshold_dec_cls= %.4f, ACC_dec= %.4f, APCER_dec= %.4f, BPCER_dec= %.4f, ACER_dec= %.4f\n
              ACC_vote= %.4f, APCER_vote= %.4f, BPCER_vote= %.4f, ACER_vote= %.4f\n""" % (
                threshold, ACC, APCER, BPCER, ACER, \
                threshold_enc_cls, ACC_cls_enc, APCER_cls_enc, BPCER_cls_enc, ACER_cls_enc, \
                threshold_depth_cls, ACC_cls_dec, APCER_cls_dec, BPCER_cls_dec, ACER_cls_dec, \
                ACC_vote, APCER_vote, BPCER_vote, ACER_vote))
        with open(os.path.join(config['save_dir'], 'ep_%03d'%ep, 'APCERs_val.txt'), 'w') as f:
            f.write('APCERs_val\n')
            for i, apcer in enumerate(APCERs_val):
                f.write('%f\n' % (apcer))
        with open(os.path.join(config['save_dir'], 'ep_%03d'%ep, 'BPCERs_val.txt'), 'w') as f:
            f.write('BPCERs_val\n')
            for i, bpcer in enumerate(BPCERs_val):
                f.write('%f\n' % (bpcer))
        with open(os.path.join(config['save_dir'], 'ep_%03d'%ep, 'APCERs_val_enc_cls.txt'), 'w') as f:
            f.write('APCERs_val_enc_cls\n')
            for i, apcer in enumerate(APCERs_val_enc_cls):
                f.write('%f\n' % (apcer))
        with open(os.path.join(config['save_dir'], 'ep_%03d'%ep, 'BPCERs_val_enc_cls.txt'), 'w') as f:
            f.write('BPCERs_val_enc_cls\n')
            for i, bpcer in enumerate(BPCERs_val_enc_cls):
                f.write('%f\n' % (bpcer))
        with open(os.path.join(config['save_dir'], 'ep_%03d'%ep, 'APCERs_val_depth_cls.txt'), 'w') as f:
            f.write('APCERs_val_depth_cls\n')
            for i, apcer in enumerate(APCERs_val_depth_cls):
                f.write('%f\n' % (apcer))
        with open(os.path.join(config['save_dir'], 'ep_%03d'%ep, 'BPCERs_val_depth_cls.txt'), 'w') as f:
            f.write('BPCERs_val_depth_cls\n')
            for i, bpcer in enumerate(BPCERs_val_depth_cls):
                f.write('%f\n' % (bpcer))
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

    ## PAD by thresholding the estimated depth map
    with open(os.path.join(save_dir, 'ACC_all_val.txt'), 'w') as f:
        f.write('ACC_all_val\n')
        for i, acc in enumerate(ACC_all_val):
            f.write('%f\n' % (acc))
    with open(os.path.join(save_dir, 'APCER_all_val.txt'), 'w') as f:
        f.write('APCER_all_val\n')
        for i, apcer in enumerate(APCER_all_val):
            f.write('%f\n' % (apcer))
    with open(os.path.join(save_dir, 'BPCER_all_val.txt'), 'w') as f:
        f.write('BPCER_all_val\n')
        for i, bpcer in enumerate(BPCER_all_val):
            f.write('%f\n' % (bpcer))
    with open(os.path.join(save_dir, 'ACER_all_val.txt'), 'w') as f:
        f.write('ACER_all_val\n')
        for i, acer in enumerate(ACER_all_val):
            f.write('%f\n' % (acer))


    ## classification by the encoder/transformer
    with open(os.path.join(save_dir, 'ACC_enc_all.txt'), 'w') as f:
        f.write('ACC_enc_all\n')
        for i, acc in enumerate(ACC_enc_all):
            f.write('%f\n' % (acc))
    with open(os.path.join(save_dir, 'APCER_enc_all.txt'), 'w') as f:
        f.write('APCER_enc_all\n')
        for i, apcer in enumerate(APCER_enc_all):
            f.write('%f\n' % (apcer))
    with open(os.path.join(save_dir, 'BPCER_enc_all.txt'), 'w') as f:
        f.write('BPCER_enc_all\n')
        for i, bpcer in enumerate(BPCER_enc_all):
            f.write('%f\n' % (bpcer))
    with open(os.path.join(save_dir, 'ACER_enc_all.txt'), 'w') as f:
        f.write('ACER_enc_all\n')
        for i, acer in enumerate(ACER_enc_all):
            f.write('%f\n' % (acer))
    with open(os.path.join(save_dir, 'threshold_all_enc_cls.txt'), 'w') as f:
        f.write('threshold_all_enc_cls\n')
        for i, threshold in enumerate(threshold_all_enc_cls):
            f.write('%f\n' % (threshold))

    ## classification based on the estimated depth map
    with open(os.path.join(save_dir, 'ACC_dec_all.txt'), 'w') as f:
        f.write('ACC_dec_all\n')
        for i, acc in enumerate(ACC_dec_all):
            f.write('%f\n' % (acc))
    with open(os.path.join(save_dir, 'APCER_dec_all.txt'), 'w') as f:
        f.write('APCER_dec_all\n')
        for i, apcer in enumerate(APCER_dec_all):
            f.write('%f\n' % (apcer))
    with open(os.path.join(save_dir, 'BPCER_dec_all.txt'), 'w') as f:
        f.write('BPCER_dec_all\n')
        for i, bpcer in enumerate(BPCER_dec_all):
            f.write('%f\n' % (bpcer))
    with open(os.path.join(save_dir, 'ACER_dec_all.txt'), 'w') as f:
        f.write('ACER_dec_all\n')
        for i, acer in enumerate(ACER_dec_all):
            f.write('%f\n' % (acer))
    with open(os.path.join(save_dir, 'threshold_all_depth_cls.txt'), 'w') as f:
        f.write('threshold_all_depth_cls\n')
        for i, threshold in enumerate(threshold_all_depth_cls):
            f.write('%f\n' % (threshold))

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

    ## classification by the encoder/transformer
    with open(os.path.join(save_dir, 'ACC_enc_all.txt'), 'w') as f:
        f.write('ACC_enc_all\n')
        for i, acc in enumerate(ACC_enc_all):
            f.write('%f\n' % (acc))
    with open(os.path.join(save_dir, 'APCER_enc_all.txt'), 'w') as f:
        f.write('APCER_enc_all\n')
        for i, apcer in enumerate(APCER_enc_all):
            f.write('%f\n' % (apcer))
    with open(os.path.join(save_dir, 'BPCER_enc_all.txt'), 'w') as f:
        f.write('BPCER_enc_all\n')
        for i, bpcer in enumerate(BPCER_enc_all):
            f.write('%f\n' % (bpcer))
    with open(os.path.join(save_dir, 'ACER_enc_all.txt'), 'w') as f:
        f.write('ACER_enc_all\n')
        for i, acer in enumerate(ACER_enc_all):
            f.write('%f\n' % (acer))
    with open(os.path.join(save_dir, 'threshold_all_enc_cls.txt'), 'w') as f:
        f.write('threshold_all_enc_cls\n')
        for i, threshold in enumerate(threshold_all_enc_cls):
            f.write('%f\n' % (threshold))

    ## classification based on the estimated depth map
    with open(os.path.join(save_dir, 'ACC_dec_all.txt'), 'w') as f:
        f.write('ACC_dec_all\n')
        for i, acc in enumerate(ACC_dec_all):
            f.write('%f\n' % (acc))
    with open(os.path.join(save_dir, 'APCER_dec_all.txt'), 'w') as f:
        f.write('APCER_dec_all\n')
        for i, apcer in enumerate(APCER_dec_all):
            f.write('%f\n' % (apcer))
    with open(os.path.join(save_dir, 'BPCER_dec_all.txt'), 'w') as f:
        f.write('BPCER_dec_all\n')
        for i, bpcer in enumerate(BPCER_dec_all):
            f.write('%f\n' % (bpcer))
    with open(os.path.join(save_dir, 'ACER_dec_all.txt'), 'w') as f:
        f.write('ACER_dec_all\n')
        for i, acer in enumerate(ACER_dec_all):
            f.write('%f\n' % (acer))
    with open(os.path.join(save_dir, 'threshold_all_depth_cls.txt'), 'w') as f:
        f.write('threshold_all_depth_cls\n')
        for i, threshold in enumerate(threshold_all_depth_cls):
            f.write('%f\n' % (threshold))

    ## classification based on the vote results
    with open(os.path.join(save_dir, 'ACC_vote_all.txt'), 'w') as f:
        f.write('ACC_vote_all\n')
        for i, acc in enumerate(ACC_vote_all):
            f.write('%f\n' % (acc))
    with open(os.path.join(save_dir, 'APCER_vote_all.txt'), 'w') as f:
        f.write('APCER_vote_all\n')
        for i, apcer in enumerate(APCER_vote_all):
            f.write('%f\n' % (apcer))
    with open(os.path.join(save_dir, 'BPCER_vote_all.txt'), 'w') as f:
        f.write('BPCER_vote_all\n')
        for i, bpcer in enumerate(BPCER_vote_all):
            f.write('%f\n' % (bpcer))
    with open(os.path.join(save_dir, 'ACER_vote_all.txt'), 'w') as f:
        f.write('ACER_vote_all\n')
        for i, acer in enumerate(ACER_vote_all):
            f.write('%f\n' % (acer))


    return

if __name__ == '__main__':

    config = json.load(open(args.config))
    config['config'] = args.config


    main_worker(config)
