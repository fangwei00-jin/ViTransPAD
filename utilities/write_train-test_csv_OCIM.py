import csv
import os
import glob
import numpy as np


file_csv_O = '/data/zming/datasets/Anti-spoof/OuluNPU/OuluNPU.csv'
file_csv_C = '/data/zming/datasets/Anti-spoof/CASIA/CASIA.csv'
file_csv_I = '/data/zming/datasets/Anti-spoof/REPLAY/REPLAY.csv'
file_csv_M = '/data/zming/datasets/Anti-spoof/MSU-MFSD/scene01/MSU.csv'
folder_csv = '/data/zming/datasets/Anti-spoof/OCIM/'
datasets = "CIM"
file_csv=os.path.join(folder_csv, "file_csv_%s.csv"%datasets)

def main():
	with open(file_csv, 'w') as cvsfile:
		writer = csv.writer(cvsfile)
		data = []
		for dataset in datasets:
			file_dataset = "file_csv_%s"%dataset
			with open(eval(file_dataset), 'r') as f:
				reader = csv.reader(f)
				for line in reader:
					data += [line]



		writer.writerows(data)


if __name__ == '__main__':
	main()
