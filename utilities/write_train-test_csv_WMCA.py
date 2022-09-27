import csv
import os
import glob
import numpy as np

videos = '/data/zming/datasets/Anti-spoof/WMCA/WMCA-Image/R_CDIT'
file_name = 'train.csv'
dataset_file = '/data/zming/datasets/Anti-spoof/WMCA/WMCA-Image/WMCA_Grandtest.txt'

# videos = '/data/zming/datasets/Anti-spoof/CASIA/test_release'
# file_name = 'test.csv'

def main():


	with open(dataset_file, 'r') as data_file:
		lines = data_file.readlines()
		bbox = '0 0 128 128'
		for line in lines[1:]:
			_,client_id, vid_name, group, label = str.split(line, ',')
			spooflabel = '0' if label[:-1]=="attack" else '1'
			vid_name = str.split(vid_name, '/')[-1]
			vid_path = os.path.join(videos, vid_name)
			modalities = os.listdir(vid_path)
			modalities.sort()
			for modality in modalities:
				print("%s"%os.path.join(vid_path, modality))
				images = glob.glob(os.path.join(vid_path, modality, '*.jpg'))
				images.sort()
				data = []
				for image in images:
					data += [[image, bbox, spooflabel]]
				with open(os.path.join(videos, "%s_%s.csv"%(group, modality)), 'a') as cvsfile:
					writer = csv.writer(cvsfile)
					writer.writerows(data)


if __name__ == '__main__':
	main()
