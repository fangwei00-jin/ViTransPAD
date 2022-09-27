<<<<<<< HEAD
<<<<<<< HEAD
import csv
import os
import glob
import numpy as np

file_train = '/data/zming/datasets/Anti-spoof/OuluNPU/Train_files/train.csv'
file_dev = '/data/zming/datasets/Anti-spoof/OuluNPU/Dev_files/dev.csv'
file_test = '/data/zming/datasets/Anti-spoof/OuluNPU/Test_files/test.csv'
file_csv = '/data/zming/datasets/Anti-spoof/OuluNPU/OuluNPU.csv'

# file_train = '/data/zming/datasets/Anti-spoof/CASIA/train_release/train.csv'
# file_test = '/data/zming/datasets/Anti-spoof/CASIA/test_release/test.csv'
# file_csv = '/data/zming/datasets/Anti-spoof/CASIA/CASIA.csv'

# file_train = '/data/zming/datasets/Anti-spoof/REPLAY/train/train.csv'
# file_dev = '/data/zming/datasets/Anti-spoof/REPLAY/devel/dev.csv'
# file_test = '/data/zming/datasets/Anti-spoof/REPLAY/test/test.csv'
# file_csv = '/data/zming/datasets/Anti-spoof/REPLAY/REPLAY.csv'
def main():
	with open(file_csv, 'w') as cvsfile:
		writer = csv.writer(cvsfile)
		data = []
		with open(file_train, 'r') as f:
			reader = csv.reader(f)
			for line in reader:
				data += [line]
		with open(file_dev, 'r') as f:
			reader = csv.reader(f)
			for line in reader:
				data += [line]
		with open(file_test, 'r') as f:
			reader = csv.reader(f)
			for line in reader:
				data += [line]



		writer.writerows(data)


if __name__ == '__main__':
	main()
=======
import csv
import os
import glob
import numpy as np

file_train = '/data/zming/datasets/Anti-spoof/OuluNPU/Train_files/train.csv'
file_dev = '/data/zming/datasets/Anti-spoof/OuluNPU/Dev_files/dev.csv'
file_test = '/data/zming/datasets/Anti-spoof/OuluNPU/Test_files/test.csv'
file_csv = '/data/zming/datasets/Anti-spoof/OuluNPU/OuluNPU.csv'

# file_train = '/data/zming/datasets/Anti-spoof/CASIA/train_release/train.csv'
# file_test = '/data/zming/datasets/Anti-spoof/CASIA/test_release/test.csv'
# file_csv = '/data/zming/datasets/Anti-spoof/CASIA/CASIA.csv'

# file_train = '/data/zming/datasets/Anti-spoof/REPLAY/train/train.csv'
# file_dev = '/data/zming/datasets/Anti-spoof/REPLAY/devel/dev.csv'
# file_test = '/data/zming/datasets/Anti-spoof/REPLAY/test/test.csv'
# file_csv = '/data/zming/datasets/Anti-spoof/REPLAY/REPLAY.csv'
def main():
	with open(file_csv, 'w') as cvsfile:
		writer = csv.writer(cvsfile)
		data = []
		with open(file_train, 'r') as f:
			reader = csv.reader(f)
			for line in reader:
				data += [line]
		with open(file_dev, 'r') as f:
			reader = csv.reader(f)
			for line in reader:
				data += [line]
		with open(file_test, 'r') as f:
			reader = csv.reader(f)
			for line in reader:
				data += [line]



		writer.writerows(data)


if __name__ == '__main__':
	main()
>>>>>>> 172a9a322294428a0719fcd0a0c808713f9124a5
=======
import csv
import os
import glob
import numpy as np

file_train = '/data/zming/datasets/Anti-spoof/OuluNPU/Train_files/train.csv'
file_dev = '/data/zming/datasets/Anti-spoof/OuluNPU/Dev_files/dev.csv'
file_test = '/data/zming/datasets/Anti-spoof/OuluNPU/Test_files/test.csv'
file_csv = '/data/zming/datasets/Anti-spoof/OuluNPU/OuluNPU.csv'

# file_train = '/data/zming/datasets/Anti-spoof/CASIA/train_release/train.csv'
# file_test = '/data/zming/datasets/Anti-spoof/CASIA/test_release/test.csv'
# file_csv = '/data/zming/datasets/Anti-spoof/CASIA/CASIA.csv'

# file_train = '/data/zming/datasets/Anti-spoof/REPLAY/train/train.csv'
# file_dev = '/data/zming/datasets/Anti-spoof/REPLAY/devel/dev.csv'
# file_test = '/data/zming/datasets/Anti-spoof/REPLAY/test/test.csv'
# file_csv = '/data/zming/datasets/Anti-spoof/REPLAY/REPLAY.csv'
def main():
	with open(file_csv, 'w') as cvsfile:
		writer = csv.writer(cvsfile)
		data = []
		with open(file_train, 'r') as f:
			reader = csv.reader(f)
			for line in reader:
				data += [line]
		with open(file_dev, 'r') as f:
			reader = csv.reader(f)
			for line in reader:
				data += [line]
		with open(file_test, 'r') as f:
			reader = csv.reader(f)
			for line in reader:
				data += [line]



		writer.writerows(data)


if __name__ == '__main__':
	main()
>>>>>>> 172a9a322294428a0719fcd0a0c808713f9124a5
