import csv
import os
import glob
import numpy as np
import cv2

#images_path = '/data/zming/datasets/writer_id/Firemaker/train'
images_path = '/data/zming/datasets/writer_id/Firemaker/test'
#file_name = 'train.csv'
file_name = 'test.csv'

images = glob.glob(os.path.join(images_path,"*.png"))
images.sort()

def main():
	with open(os.path.join(images_path, file_name), 'w') as cvsfile:
		writer = csv.writer(cvsfile)
		train_data = []
		frame_num = 0
		totalframes = len(images)
		for img_file in images:
			img=cv2.imread(img_file)
			h, w, c = img.shape
			line = '0'+ ' '+'0'+' '+'%d'%w+' '+'%d'%h #(left-top-x,left-top-y,right-bottom-x,right-bottom-y)
			img_name = str.split(img_file, "/")[-1]
			writer_id = str.split(img_name, "-")[0]
			train_data += [[img_file, line, writer_id]]
			frame_num += 1
			print("%d/%d"%(frame_num, totalframes))

		writer.writerows(train_data)

if __name__ == '__main__':
	main()
