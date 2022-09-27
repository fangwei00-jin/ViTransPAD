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
from pytorch_grad_cam.utils.image import show_cam_on_image, \
	preprocess_image


def get_args():
	parser = argparse.ArgumentParser()
	parser.add_argument('--use-cuda', action='store_true', default=False,
						help='Use NVIDIA GPU acceleration')
	parser.add_argument(
		'--image-path',
		type=str,
		default='./examples/both.png',
		help='Input image path')
	parser.add_argument('--aug_smooth', action='store_true',
						help='Apply test time augmentation to smooth the CAM')
	parser.add_argument(
		'--eigen_smooth',
		action='store_true',
		help='Reduce noise by taking the first principle componenet'
		'of cam_weights*activations')

	parser.add_argument(
		'--method',
		type=str,
		default='gradcam',
		help='Can be gradcam/gradcam++/scorecam/xgradcam/ablationcam')
	parser.add_argument('-c', '--config', type=str)

	args = parser.parse_args()
	args.use_cuda = args.use_cuda and torch.cuda.is_available()
	if args.use_cuda:
		print('Using GPU for acceleration')
	else:
		print('Using CPU for computation')

	return args





if __name__ == '__main__':
	""" python vit_gradcam.py -image-path <path_to_image>
	Example usage of using cam-methods on a VIT network.

	"""

	args = get_args()
	methods = \
		{"gradcam": GradCAM,
		 "scorecam": ScoreCAM,
		 "gradcam++": GradCAMPlusPlus,
		 "ablationcam": AblationCAM,
		 "xgradcam": XGradCAM,
		 "eigencam": EigenCAM,
		 "eigengradcam": EigenGradCAM,
		 "layercam": LayerCAM,
		 "fullgrad": FullGrad}

	if args.method not in list(methods.keys()):
		raise Exception(f"method should be one of {list(methods.keys())}")


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


		# model = torch.hub.load('facebookresearch/deit:main',
		#                        'deit_tiny_patch16_224', pretrained=True)

		model.eval()
		for name, p in model.named_parameters():
			print(name, p.size())

		if args.use_cuda:
			model = model.cuda()

		if args.method not in methods:
			raise Exception(f"Method {args.method} not implemented")


		######## Vit ###############
		# def reshape_transform(tensor, height=14, width=14):
		#
		#     result=tensor[:,1:,:].reshape(tensor.size(0),
		#                                     height, width, tensor.size(2))
		#     # Bring the channels to the first dimension, like in CNNs.
		#     result=result.transpose(2, 3).transpose(1, 2)
		#     return result
		#
		# #target_layers_str = "model.encoder.norm"
		# target_layers_str = "model.encoder.blocks[-1].norm1"
		# target_layers = [eval(target_layers_str)]
		# cam = methods[args.method](model=model,
		#                            target_layers=target_layers,
		#                            use_cuda=args.use_cuda,
		#                            reshape_transform=reshape_transform)

		######## Efficientnet ###############
		# target_layers_str = "model.encoder._conv_head"
		# target_layers = [eval(target_layers_str)]
		# cam = methods[args.method](model=model,
		#                            target_layers=target_layers,
		#                            use_cuda=args.use_cuda)



		# ######### TransFAS ###############
		# ## Encoder Efficientnet conv layer
		# target_layers_str = "model.encoder._conv_head"
		# target_layers=[eval(target_layers_str)]
		# cam = methods[args.method](model=model,
		#                            target_layers=target_layers,
		#                            use_cuda=args.use_cuda)

		# # ## Encoder vit last layer
		# def reshape_transform(tensor, height=14, width=14):
		#
		#     result=tensor[:,1:,:].reshape(tensor.size(0),
		#                                     height, width, tensor.size(2))
		#     # Bring the channels to the first dimension, like in CNNs.
		#     result=result.transpose(2, 3).transpose(1, 2)
		#     return result
		#
		# #target_layers_str = "model.encoder.norm"
		# target_layers_str = "model.encoder.blocks[-1].norm1"
		# target_layers = [eval(target_layers_str)]
		# cam = methods[args.method](model=model,
		#                            target_layers=target_layers,
		#                            use_cuda=args.use_cuda,
		#                            reshape_transform=reshape_transform)

		# ## Transformer Q, K, V embedding conv layer ##
		# target_layers_str = "model.transformer[0].attention.value_embedding"
		# target_layers=[eval(target_layers_str)]
		# print(target_layers[0])
		# cam = methods[args.method](model=model,
		#                            target_layers=target_layers,
		#                            use_cuda=args.use_cuda)

		# ## Transformer attention results concatenation conv layer ##
		# target_layers_str = "model.transformer[0].attention.output_linear[0]"
		# target_layers=[eval(target_layers_str)]
		# print(target_layers[0])
		# cam = methods[args.method](model=model,
		#                            target_layers=target_layers,
		#                            use_cuda=args.use_cuda)


		# Transformer feed forward convs layer ##
		target_layers_str = "model.transformer[3].feed_forward.conv[2]"
		target_layers=[eval(target_layers_str)]
		print(target_layers[0])
		cam = methods[args.method](model=model,
								   target_layers=target_layers,
								   use_cuda=args.use_cuda)

		# ## fc layer ###
		# def reshape_transform(tensor, height=32, width=32):
		#     if len(tensor.size()) == 2:
		#         tensor = tensor[:, :, None, None]
		#     result=tensor.reshape(tensor.size(0),
		#                                     height, width, tensor.size(2))
		#     # Bring the channels to the first dimension, like in CNNs.
		#     result=result.transpose(2, 3).transpose(1, 2)
		#     return result
		# target_layers_str = "model.fc1"
		# target_layers=[eval(target_layers_str)]
		# print(target_layers[0])
		# cam = methods[args.method](model=model,
		#                            target_layers=target_layers,
		#                            use_cuda=args.use_cuda,
		#                            reshape_transform=reshape_transform)

		##########################################################


		#### read the images ###
		for batch_idx, batch in enumerate(trainer.test_loader):
			frames, masks, spoofing_labels, images_path, images=batch['frame_tensors'].to(device), batch['mask_tensors'].to(device), \
														batch['spoofing_labels'], batch['images_path'], batch['frames']
			# frames, masks, spoofing_labels=batch['frame_tensors'].to(device), batch['mask_tensors'].to(device), \
			#                                             batch['spoofing_labels']

			vid_name=str.split(images_path[0][0],'/')[-2]
			vid_dir = os.path.join(".",model_dir, vid_name)
			if not os.path.isdir(vid_dir):
				os.mkdir(vid_dir)
			imgs_name = []
			for img_path in images_path:
				print(img_path)
				strs_img_path = str.split(img_path[0],'/')
				imgs_name.append(strs_img_path[-1][:-4])


			target_category = None
			#target_category = [spoofing_labels.item()]
			#target_category = [0]

			# AblationCAM and ScoreCAM have batched implementations.
			# You can override the internal batch size for faster computation.
			#cam.batch_size = 32
			cam.batch_size = len(batch)

			input_tensor= frames
			mask_tensor = masks
			grayscale_cams, pred_labels = cam(input_tensor=input_tensor,
								mask_tensor=mask_tensor,
								target_category=target_category,
								eigen_smooth=args.eigen_smooth,
								aug_smooth=args.aug_smooth)

			print("Ground truth: %s"%spoofing_labels)
			print("Attention target category: %s"%target_category)
			print("Predicted results: %s" % pred_labels)
			with open('./%s/target_category.txt'%vid_dir, 'w') as camfile:
				camfile.write("Ground truth: %s\n"%spoofing_labels)
				camfile.write("Attention target category: %s\n"%target_category)
				camfile.write("Predicted results by model: %s\n" % pred_labels)
				camfile.write("target_layers_name: %s\n" % target_layers_str)
				camfile.write("target_layers: %s\n" % target_layers)

			# Here grayscale_cam has only one image in the batch
			#grayscale_cam = grayscale_cam[0, :]
			for i, grayscale_cam in enumerate(grayscale_cams):
				rgb_img = images[i][0].numpy()
				rgb_img = rgb_img[:,:,::-1]  ## PIL format to cv2 format
				rgb_img=np.float32(rgb_img) / 255
				cam_image = show_cam_on_image(rgb_img, grayscale_cam)
				cv2.imwrite(f'./%s/{args.method}_cam_%s.jpg'%(vid_dir, imgs_name[i]), cam_image)

			# # Here grayscale_cam has only one image in the batch
			# grayscale_cam = grayscale_cam[0, :]
			#
			# cam_image = show_cam_on_image(rgb_img, grayscale_cam)
			# cv2.imwrite(f'{args.method}_cam.jpg', cam_image)

			print('===========================\n')