import os
import glob
import numpy as np
from sklearn.metrics import roc_curve, auc
import cv2
import matplotlib
#matplotlib.use('Agg')
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
images_path = '/data/zming/logs/Antispoof/test/SiW/Transfomer_FAS/protocol1/1layer_4head_5frames_threshold/val-train_test/112-117/0112/val_images'

images = glob.glob(os.path.join(images_path, '*_pre.jpg'))
images.sort()

def get_err_threhold(fpr, tpr, threshold):
    # RightIndex = (tpr + (1 - fpr) - 1);
    # right_index = np.argmax(RightIndex)
    # best_th = threshold[right_index]
    # err = fpr[right_index]

    # differ_tpr_fpr_1 = tpr + fpr - 1.0  ## best threshold make tpr -> 1 and fpr -> 0, thus tpr+fpr-1->0
    differ_tpr_fpr_1 = 1.0 - tpr + fpr  ## best threshold make tpr -> 1 and fpr -> 0, thus tpr+fpr-1->0
    # right_index = np.argmin(differ_tpr_fpr_1)  ## this is the upper bound  of the threshold based on the current dataset
    idx = np.argsort(differ_tpr_fpr_1)
    threshold_upperbound, threshold_lowerbound = threshold[idx[0]], threshold[idx[0] + 1]
    # print("Threshold[right_index]: %f right index %d" % (threshold[right_index], right_index))

    # right_index = np.argmin(fpr ** 2 + (1 - tpr) ** 2)  ## Best threshold
    # print("Threshold[right_index]: %f right index %d" % (threshold[right_index], right_index))

    # best_th = threshold[right_index]
    # err = fpr[right_index]

    # print(err, best_th)
    # return err, best_th
    return threshold_upperbound, threshold_lowerbound

score_list = []
spoof_list = []
for image_path in images:
    image_name = str.split(image_path, '/')[-1]
    image_name = str.split(image_name, '_')[0]
    type_id = str.split(image_name, '-')[2]
    if type_id == '1':
        spoofing_label = 1
    else:
        spoofing_label = 0

    image= cv2.imread(image_path)
    histogram, bin_edges = np.histogram(image, bins=256, range=(0, 255))
    # idx = np.argmax(histogram)
    # if 129>bin_edges[idx]>120 and histogram[idx]/np.sum(histogram)>0.9:
    # idx = [61, 62, 63, 64] ## bin_edges in [120, 129]
    idx = list(range(100, 151))  ## bin_edges in [100, 140]
    if np.sum(histogram[idx] / np.sum(histogram)) > 0.80:  ## more than 70% pixels are in [120,129]
        score = max(0, np.mean(image - 128.0) / 255.0)
        if spoofing_label == 1:
            print("False Attack +++++++++ %s"%image_path)
    else:
        # print(image_path)
        score = np.mean(image) / 255.0
        if spoofing_label == 0:
            print("False Livness ============== %s"%image_path)
    score_list.append(score)
    spoof_list.append(spoofing_label)

fpr, tpr, thresholds = roc_curve(spoof_list, score_list, pos_label=1, drop_intermediate=False)
threshold_upperbound, threshold_lowerbound = get_err_threhold(fpr, tpr, thresholds)
threshold = (threshold_upperbound + threshold_lowerbound) / 2