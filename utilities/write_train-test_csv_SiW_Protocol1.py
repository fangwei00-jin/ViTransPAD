import csv
import os
import glob


videos_train = '/data/zming/datasets/SiW/SiW_release/Train'
file_name = 'train_protocol1.csv'
# videos_train = '/data/zming/datasets/SiW/SiW_release/Test'
# file_name = 'test_protocol1.csv'

def main():
	with open(os.path.join(videos_train, file_name), 'w') as cvsfile:
		writer = csv.writer(cvsfile)
		folders = os.listdir(videos_train)
		folders.sort()
		i_max = 0

		for folder in folders:
			if not os.path.isdir(os.path.join(videos_train, folder)):
				continue
			spooflabel = 1 if folder=='live' else 0
			subjects = os.listdir(os.path.join(videos_train, folder))
			subjects.sort()
			for sub in subjects:
				print('%s %d'%(sub, i_max))
				face_files = glob.glob(os.path.join(videos_train, folder, sub, '*.face'))
				face_files.sort()				
				for file in face_files: ## one file for one video
					train_data = []
					frame_num = 0
					with open(file, 'r') as f:
						lines = f.readlines();
						i = len(lines)
						i_max = i if i > i_max else i_max
						for i, line in enumerate(lines):
							# if i==406:
							# 	print('%s %d' % (file, i))
							if line == '0 0 0 0 \n':
								print('%s %d'%(file, i))
								continue
							elif not line: ## line is empty
								print('%s %d is empty'%(file, i))
								continue
							elif frame_num == 60: ### only for train dataset
								break
							else:
								train_data += [[file[:-5]+'/%04d.jpg'%(i+1), line[:-1], spooflabel]]
								frame_num += 1

					writer.writerows(train_data)



if __name__ == '__main__':
	main()
