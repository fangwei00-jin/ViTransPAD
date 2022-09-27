import csv
import os
import glob
import pandas as pd
import numpy as np

# videos_file_old = '/data/zming/datasets/Anti-spoof/OuluNPU/Train_files/train_protocol1_old.csv'
# videos_file = '/data/zming/datasets/Anti-spoof/OuluNPU/Train_files/train_protocol1.csv'
# videos_file_old= '/data/zming/datasets/Anti-spoof/OuluNPU/Test_files/test_protocol1_old.csv'
# videos_file= '/data/zming/datasets/Anti-spoof/OuluNPU/Test_files/test_protocol1.csv'
# videos_file_old= '/data/zming/datasets/SiW/SiW_release/Train/train_protocol3_type13_old.csv'
# videos_file= '/data/zming/datasets/SiW/SiW_release/Train/train_protocol3_type13.csv'
videos_file_old= '/data/zming/datasets/SiW/SiW_release/Test/test_protocol3_type12_old.csv'
videos_file= '/data/zming/datasets/SiW/SiW_release/Test/test_protocol3_type12.csv'

def main():
	df_video = pd.read_csv(videos_file_old, sep=',', index_col=None, header=None,
				names=['image_path', 'bbox', 'spoofing_label'])

	image_paths = df_video['image_path']
	spoofing_labels = df_video['spoofing_label']
	bboxes = df_video['bbox']
	error_bbox = []
	## check bbox ##
	for idx, bbox in enumerate(bboxes):
		bbox = bboxes[idx]
		bbox_num = [int(p) for p in str.split(bbox, " ") if (p.startswith('-') and p[1:] or p).isdigit()]

		## check bbox if all coordinates are 0 as [0 0 0 0]
		if not any(np.array(bbox_num)):  ## if there is no a non zero element
			print("bbox is empty in %s!" % image_paths[idx])
			error_bbox.append(idx)

		# check bbox if having 4 coordinates of top-left, bottom-right
		if len(bbox_num) < 4:
			print("bbox has coordinates less than 4 in %s!" % image_paths[idx])
			error_bbox.append(idx)

	## write the new csv file
	train_data = []
	with open(videos_file, 'w') as cvsfile:
		writer = csv.writer(cvsfile)
		for idx, bbox in enumerate(bboxes):
			if idx not in error_bbox:
				train_data += [[image_paths[idx], bboxes[idx], spoofing_labels[idx]]]

		writer.writerows(train_data)


if __name__ == '__main__':
	main()
