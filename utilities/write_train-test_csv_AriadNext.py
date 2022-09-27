<<<<<<< HEAD
<<<<<<< HEAD
import csv
import os
import glob
import numpy as np

#videos = '/data/zming/datasets/Anti-spoof/CASIA/train_release'
videos = '/data/zming/datasets/Anti-spoof/SELFCOLLECT_Data/AriadNext/Dataset_PAD_L3i_v1_14_04_2021/train'
file_name = 'train.csv'

# videos = '/data/zming/datasets/Anti-spoof/CASIA/test_release'
# file_name = 'test.csv'

def main():
	with open(os.path.join(videos, file_name), 'w') as cvsfile:
		writer = csv.writer(cvsfile)
		types = os.listdir(videos)
		types.sort()
		for type in types:
			subs = os.listdir(os.path.join(videos, type))
			subs.sort()
			i_max = 0


			for sub in subs:## one folder for one video
				if not os.path.isdir(os.path.join(videos, type, sub)):
					continue
				folders = os.listdir(os.path.join(videos, type, sub))
				folders.sort()
				for folder in folders:
					if not os.path.isdir(os.path.join(videos, type, sub, folder)):
						continue

					spooflabel = 1 if type == 'Legits' else 0


					print (os.path.join(videos, sub, folder))
					face_files = glob.glob(os.path.join(videos, type, sub, folder, '*_bbox.txt'))
					face_files.sort()
					train_data = []
					for file in face_files:

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
								## bbox length less than 4
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
								train_data += [[file[:-9]+'.jpg', line, spooflabel]]
								frame_num += 1

					# train_data_resample = []
					# if spooflabel == 1:
					# 	## to balance the live video with the attack video since attack:live is 3:1
					# 	for i in range(3):
					# 		print(i)
					# 		for l in train_data:
					# 			img_name = str.split(l[0], '/')[-1]
					# 			img_path = os.path.join(videos, sub, folder+'_%d'%i, img_name)
					# 			train_data_resample.append([img_path, l[1], l[2]])
					# else:
					# 	for i in range(1):
					# 		for l in train_data:
					# 			img_name = str.split(l[0], '/')[-1]
					# 			img_path = os.path.join(videos, sub, folder+'_%d'%i, img_name)
					# 			l[0] = img_path
					# 			train_data_resample.append(l)
					#
					# writer.writerows(train_data_resample)
	#
					writer.writerows(train_data)

if __name__ == '__main__':
	main()
=======
import csv
import os
import glob
import numpy as np

#videos = '/data/zming/datasets/Anti-spoof/CASIA/train_release'
videos = '/data/zming/datasets/Anti-spoof/SELFCOLLECT_Data/AriadNext/Dataset_PAD_L3i_v1_14_04_2021/train'
file_name = 'train.csv'

# videos = '/data/zming/datasets/Anti-spoof/CASIA/test_release'
# file_name = 'test.csv'

def main():
	with open(os.path.join(videos, file_name), 'w') as cvsfile:
		writer = csv.writer(cvsfile)
		types = os.listdir(videos)
		types.sort()
		for type in types:
			subs = os.listdir(os.path.join(videos, type))
			subs.sort()
			i_max = 0


			for sub in subs:## one folder for one video
				if not os.path.isdir(os.path.join(videos, type, sub)):
					continue
				folders = os.listdir(os.path.join(videos, type, sub))
				folders.sort()
				for folder in folders:
					if not os.path.isdir(os.path.join(videos, type, sub, folder)):
						continue

					spooflabel = 1 if type == 'Legits' else 0


					print (os.path.join(videos, sub, folder))
					face_files = glob.glob(os.path.join(videos, type, sub, folder, '*_bbox.txt'))
					face_files.sort()
					train_data = []
					for file in face_files:

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
								## bbox length less than 4
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
								train_data += [[file[:-9]+'.jpg', line, spooflabel]]
								frame_num += 1

					# train_data_resample = []
					# if spooflabel == 1:
					# 	## to balance the live video with the attack video since attack:live is 3:1
					# 	for i in range(3):
					# 		print(i)
					# 		for l in train_data:
					# 			img_name = str.split(l[0], '/')[-1]
					# 			img_path = os.path.join(videos, sub, folder+'_%d'%i, img_name)
					# 			train_data_resample.append([img_path, l[1], l[2]])
					# else:
					# 	for i in range(1):
					# 		for l in train_data:
					# 			img_name = str.split(l[0], '/')[-1]
					# 			img_path = os.path.join(videos, sub, folder+'_%d'%i, img_name)
					# 			l[0] = img_path
					# 			train_data_resample.append(l)
					#
					# writer.writerows(train_data_resample)
	#
					writer.writerows(train_data)

if __name__ == '__main__':
	main()
>>>>>>> 172a9a322294428a0719fcd0a0c808713f9124a5
=======
import csv
import os
import glob
import numpy as np

#videos = '/data/zming/datasets/Anti-spoof/CASIA/train_release'
videos = '/data/zming/datasets/Anti-spoof/SELFCOLLECT_Data/AriadNext/Dataset_PAD_L3i_v1_14_04_2021/train'
file_name = 'train.csv'

# videos = '/data/zming/datasets/Anti-spoof/CASIA/test_release'
# file_name = 'test.csv'

def main():
	with open(os.path.join(videos, file_name), 'w') as cvsfile:
		writer = csv.writer(cvsfile)
		types = os.listdir(videos)
		types.sort()
		for type in types:
			subs = os.listdir(os.path.join(videos, type))
			subs.sort()
			i_max = 0


			for sub in subs:## one folder for one video
				if not os.path.isdir(os.path.join(videos, type, sub)):
					continue
				folders = os.listdir(os.path.join(videos, type, sub))
				folders.sort()
				for folder in folders:
					if not os.path.isdir(os.path.join(videos, type, sub, folder)):
						continue

					spooflabel = 1 if type == 'Legits' else 0


					print (os.path.join(videos, sub, folder))
					face_files = glob.glob(os.path.join(videos, type, sub, folder, '*_bbox.txt'))
					face_files.sort()
					train_data = []
					for file in face_files:

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
								## bbox length less than 4
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
								train_data += [[file[:-9]+'.jpg', line, spooflabel]]
								frame_num += 1

					# train_data_resample = []
					# if spooflabel == 1:
					# 	## to balance the live video with the attack video since attack:live is 3:1
					# 	for i in range(3):
					# 		print(i)
					# 		for l in train_data:
					# 			img_name = str.split(l[0], '/')[-1]
					# 			img_path = os.path.join(videos, sub, folder+'_%d'%i, img_name)
					# 			train_data_resample.append([img_path, l[1], l[2]])
					# else:
					# 	for i in range(1):
					# 		for l in train_data:
					# 			img_name = str.split(l[0], '/')[-1]
					# 			img_path = os.path.join(videos, sub, folder+'_%d'%i, img_name)
					# 			l[0] = img_path
					# 			train_data_resample.append(l)
					#
					# writer.writerows(train_data_resample)
	#
					writer.writerows(train_data)

if __name__ == '__main__':
	main()
>>>>>>> 172a9a322294428a0719fcd0a0c808713f9124a5
