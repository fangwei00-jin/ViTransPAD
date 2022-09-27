import csv
import os
import glob
import numpy as np

# videos = '/data/zming/datasets/Anti-spoof/OuluNPU/Train_files'
# file_name = 'train_protocol2.csv'
videos = '/data/zming/datasets/Anti-spoof/OuluNPU/Dev_files'
file_name = 'dev_protocol2.csv'
Files = [1, 2, 4]

# videos = '/data/zming/datasets/Anti-spoof/OuluNPU/Test_files'
# file_name = 'test_protocol2.csv'
# Files = [1, 3, 5]

def main():
	with open(os.path.join(videos, file_name), 'w') as cvsfile:
		writer = csv.writer(cvsfile)
		folders = os.listdir(videos)
		folders.sort()
		i_max = 0

		for folder in folders:
			if not os.path.isdir(os.path.join(videos, folder)):
				continue
			folderstr = str.split(folder, '_')
			Phone, Session, User, File = folderstr[0], folderstr[1], folderstr[2], folderstr[3]

			spooflabel = 1 if File=='1' else 0

			if int(File) not in Files:
				continue
			print (folder)
			face_files = glob.glob(os.path.join(videos, folder, '*_bbox.txt'))
			face_files.sort()
			train_data = []
			for file in face_files: ## one file for one video

				frame_num = 0
				with open(file, 'r') as f:
					lines = f.readlines()
					i = len(lines)
					i_max = i if i > i_max else i_max
					for j, line in enumerate(lines):
						## line is empty
						if not line:
							print('%s %d bbox is empty'%(file, j))
							##line = '0 0 0 0'
							continue
						## bbox's length less than 4
						if len([p for p in line if (p.startswith('-') and p[1:] or p).isdigit()]) < 4:
							print('%s %d has not enough bbox coordinates' % (file, j))
							#line = '0 0 0 0'
							continue
						## bbox are all zero [0 0 0 0]
						bbox_num = [int(p) for p in str.split(line, " ") if
									(p.startswith('-') and p[1:] or p).isdigit()]
						if not any(np.array(bbox_num)):  ## if there is no a non zero element
							print("bbox is empty in %s!" % file)
							continue
						bboxstr = str.split(line, ' ')
						line = bboxstr[0]+' '+bboxstr[2]+' '+bboxstr[1]+' '+bboxstr[3]
						train_data += [[file[:-9]+'.jpg', line[:-1], spooflabel]]
						frame_num += 1

			train_data_resample = []
			if spooflabel == 1:
				## to balance the live video with the attack video since attack:live is 4:1
				for i in range(2):
					print(i)
					for l in train_data:
						img_name = str.split(l[0], '/')[-1]
						img_path = os.path.join(videos, folder+'_%d'%i, img_name)
						train_data_resample.append([img_path, l[1], l[2]])
			else:
				for i in range(1):
					for l in train_data:
						img_name = str.split(l[0], '/')[-1]
						img_path = os.path.join(videos, folder+'_%d'%i, img_name)
						l[0] = img_path
						train_data_resample.append(l)

			writer.writerows(train_data_resample)


if __name__ == '__main__':
	main()
