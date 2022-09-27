import argparse
import cv2
import numpy as np
import torch
import sys
import json
import importlib
from PIL import Image
import os
from shutil import copy
import glob
sys.path.append('../..')
def get_args():
	parser = argparse.ArgumentParser()

	parser.add_argument('-c', '--config', type=str)

	args = parser.parse_args()


	return args

if __name__ == '__main__':

	args=get_args()
	config = json.load(open(args.config))
	copy(args.config, './')
	config['config']=args.config
	config['distributed']=False
	config['world_size']=1
	config['init_method']='tcp://localhost:12345'
	config['global_rank']=0
	config['local_rank']=0
	if torch.cuda.is_available():
		config['device'] = torch.device("cuda:{}".format(config['local_rank']))
	else:
		config['device'] = 'cpu'
	device=config['device']

	Trainer = importlib.import_module(config['train']['train_module'])

	models_dir = config['train']['pretrained_model'][:-13]

	models = glob.glob(os.path.join(models_dir,'gen_*.pth'))

	for model in models:
		model_name = str.split(model, '/')[-1][:-4]
		model_dir = os.path.join(".",model_name)
		if not os.path.isdir(model_dir):
			os.mkdir(model_dir)

		config['train']['pretrained_model'] = model

		trainer = Trainer.Trainer(config, debug=True)

		model = trainer.netG

		model.eval()
		# for name, p in model.named_parameters():
		#     print(name, p.size())



		#### read the images ###
		gt_labels_all = []
		pred_labels_all = []
		for batch_idx, batch in enumerate(trainer.test_loader):
			frames, masks, spoofing_labels, images_path =batch['frame_tensors'].to(device), batch['mask_tensors'].to(device), \
														batch['spoofing_labels'], batch['images_path']
			print("batch %d/%d"%(batch_idx, len(trainer.test_loader)))
			output = model(frames, masks)
			pred_labels=np.argmax(output.cpu().data.numpy(), axis=-1)

			t = frames.size()[1]
			labels = [[spoofing_label.item()]*t for spoofing_label in spoofing_labels]
			labels = [item for sublist in labels for item in sublist]
			gt_labels_all += labels
			pred_labels_all += list([item for item in pred_labels])

		### Evaluation acc, apcer, bpcer, acer
		gt_labels_all = np.array(gt_labels_all)
		pred_labels_all = np.array(pred_labels_all)
		results = (gt_labels_all==pred_labels_all)
		acc = results.sum()/len(gt_labels_all)
		false_positive_err = [result for i, result in enumerate(results) if result==False and gt_labels_all[i]==0]
		false_negative_err = [result for i, result in enumerate(results) if result==False and gt_labels_all[i]==1]
		negative_elements = [gt for gt in gt_labels_all if gt==0]
		positive_elements=[gt for gt in gt_labels_all if gt == 1]
		apcer = len(false_positive_err) / len(negative_elements)
		bpcer = len(false_negative_err) / len(positive_elements)
		acer = (apcer+bpcer)/2.0

		print("Predicted acc: %f" % acc)
		print("Predicted apcer: %f" % apcer)
		print("Predicted bpcer: %f" % bpcer)
		print("Predicted acer: %f" % acer)
		with open('./%s/acc.txt'%model_dir, 'w') as camfile:
			camfile.write("Predicted acc: %f\n" % acc)
			camfile.write("Predicted apcer: %f\n" % apcer)
			camfile.write("Predicted bpcer: %f\n" % bpcer)
			camfile.write("Predicted acer: %f\n" % acer)

