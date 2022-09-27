import csv
import os
import glob
import numpy as np

videos = '/data/zming/datasets/Anti-spoof/REPLAY/train'
file_name = 'train_resample.csv'
# videos = '/data/zming/datasets/Anti-spoof/REPLAY/devel'
# file_name = 'dev.csv'

# videos = '/data/zming/datasets/Anti-spoof/REPLAY/test'
# file_name = 'test.csv'

def main():
	with open(os.path.join(videos, file_name), 'w') as cvsfile:
		writer = csv.writer(cvsfile)
		types = os.listdir(videos)
		types.sort()
		i_max = 0

		for type in types:
			#type = 'real'  # debug
			if not os.path.isdir(os.path.join(videos, type)):
				continue
			if type == 'attack':
				sessions = os.listdir(os.path.join(videos, type))
				sessions.sort()
				prefolders = [os.path.join(videos, type, session) for session in sessions]
			else:
				prefolders = [os.path.join(videos, type)]


			for prefolder in prefolders:
				spooflabel = 1 if str.split(prefolder, '/')[-1] == 'real' else 0

				folders = os.listdir(prefolder)
				folders.sort()
				for folder in folders:
					if not os.path.isdir(os.path.join(prefolder, folder)):
						continue


					print (os.path.join(prefolder, folder))
					face_files = glob.glob(os.path.join(prefolder, folder, '*_bbox.txt'))
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
								train_data += [[file[:-9]+'.jpg', line[:-1], spooflabel]]
								frame_num += 1

					train_data_resample = []
					if spooflabel == 1:
						## to balance the live video with the attack video since attack:live is 5:1
						for i in range(5):
							print(i)
							for l in train_data:
								img_name = str.split(l[0], '/')[-1]
								img_path = os.path.join(prefolder, folder+'_%d'%i, img_name)
								train_data_resample.append([img_path, l[1], l[2]])
					else:
						for i in range(1):
							for l in train_data:
								img_name = str.split(l[0], '/')[-1]
								img_path = os.path.join(prefolder, folder+'_%d'%i, img_name)
								l[0] = img_path
								train_data_resample.append(l)

					writer.writerows(train_data_resample)
					# writer.writerows(train_data)
#

if __name__ == '__main__':
	main()
