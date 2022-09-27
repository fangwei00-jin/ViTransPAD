import csv
import os
import glob
import numpy as np
import pandas as pd

# video streams Parameters
# Idiap's session_id
# Idiap's Attack types and PAIs
BATL_CONFIG = {0: {'name': 'Bona Fide',
		   'pai': {0: 'Unknown, id: 000',
			   1: 'Client own medical glasses',
			   2: 'Unisex glasses G-01, id: 511'}},
		   1: {'name': 'Facial disguise',
		   'pai': {0: 'Unknown, id: 000',
			   1: 'Funny eyes glasses G-02, id: 512',
			   2: 'Plastic halloween mask B-08-22-00, id: 101',
			   3: 'unassigned',
			   4: 'Funny eyes glasses G-03, id: 513',
			   5: 'Funny eyes glasses G-04, id: 514',
			   6: 'Funny eyes glasses G-05, id: 515',
			   7: 'Funny eyes glasses G-06, id: 516',
			   8: 'Wig W-01, id: 520',
			   9: 'Wig W-02, id: 521',
			   10: 'Wig W-03, id: 522',
			   11: 'Wig W-04, id: 523',
			   12: 'Wig W-05, id: 524',
			   13: 'Exotic femail transparent mask B08-532-00, id: 532',
			   14: 'Exotic femail transparent mask with make up B08-23-01, id: 533',
			   15: 'Paper glasses G-10, id: 534',
			   16: 'Paper glasses G-11, id: 535'}},
		   2: {'name': 'Fake face',
		   'pai': {0: 'Unknown, id: 000',
			   1: 'Mannequin Mq-07, id: 507',
			   2: 'Mannequin Mq-01, id: 501',
			   3: 'Mannequin Mq-02, id: 502',
			   4: 'Mannequin Mq-03, id: 503',
			   5: 'Mannequin Mq-04, id: 504',
			   6: 'Mannequin Mq-05, id: 505',
			   7: 'Mannequin Mq-06, id: 506',
			   8: 'Papercraft mask B03-01-00, id: 001',
			   9: 'Papercraft mask B03-03-00, id: 099',
			   10: 'Papercraft mask B03-11-00, id: 100',
			   11: 'Papercraft mask B03-12-00, id: 012',
			   12: 'Custom wearable mask B05-01-00, id: 001',
			   13: 'Custom wearable mask B05-02-00, id: 002',
			   14: 'Custom wearable mask B05-12-00, id: 012',
			   15: 'Custom wearable mask B05-16-00, id: 016',
			   16: 'Custom wearable mask B05-18-00, id: 018',
			   17: 'Custom wearable mask B05-19-00, id: 019',
			   18: 'Flexible silicon mask B07-21-00, id: 530',
			   75: 'Flexible silicon mask B06-30-00, id: 064',
			   76: 'Flexible silicon mask B06-31-00, id: 107',
			   77: 'Flexible silicon mask B07-32-00, id: 108',
			   78: 'Flexible silicon mask B07-33-00, id: 109',
			   79: 'Flexible silicon mask B07-34-00, id: 110',
			   80: 'Flexible silicon mask B06-01-00, id: 001',
			   81: 'Flexible silicon mask B06-02-00, id: 002',
			   82: 'Flexible silicon mask B06-02-02, id: 002',
			   83: 'Flexible silicon mask B06-12-00, id: 012',
			   84: 'Flexible silicon mask B06-24-00, id: 066',
			   85: 'Flexible silicon mask B06-16-00, id: 016',
			   86: 'Flexible silicon mask B06-18-00, id: 018',
			   87: 'Flexible silicon mask B06-19-00, id: 019',
			   88: 'Flexible silicon mask B07-01-00, id: 001',
			   89: 'Flexible silicon mask B07-12-00, id: 012',
			   90: 'Flexible silicon mask B07-19-00, id: 019',
			   91: 'Flexible silicon mask B06-25-00, id: 102',
			   92: 'Flexible silicon mask B06-26-01, id: 103',
			   93: 'Flexible silicon mask B07-27-01, id: 104',
			   94: 'Flexible silicon mask B07-28-01, id: 105',
			   95: 'Flexible silicon mask B07-29-01, id: 106',
			   96: 'Flexible silicon mask B06-26-00, id: 103',
			   97: 'Flexible silicon mask B07-27-00, id: 104',
			   98: 'Flexible silicon mask B07-28-00, id: 105',
			   99: 'Flexible silicon mask B07-29-00, id: 106'}},
		   3: {'name': 'Printed',
		   'pai': {0: 'Unknown, id: 000',
			   1: 'Printed P00-001-00, id: 001',
			   2: 'Printed P00-001-01, id: 001',
			   3: 'Printed P00-001-02, id: 001',
			   4: 'Printed P00-001-03, id: 001',
			   5: 'Electronic P01-001-00, id: 001',
			   6: 'Printed P00-018-00, id: 018',
			   7: 'Printed P00-018-01, id: 018',
			   8: 'Printed P00-018-02, id: 018',
			   9: 'Printed P00-018-03, id: 018',
			   10: 'Electronic P01-018-00, id: 018',
			   11: 'Printed P00-098-00, id: 098',
			   12: 'Printed P00-098-01, id: 098',
			   13: 'Printed P00-098-02, id: 098',
			   14: 'Printed P00-098-03, id: 098',
			   15: 'Electronic P01-098-00, id: 098'}},
		   4: {'name': 'Video',
		   'pai': {0: 'Unknown, id: 000',
			   1: 'Video play V00-019-00, id: 019',
			   2: 'Video pause V01-019-00, id: 019',
			   3: 'Video play V00-012-00, id: 012',
			   4: 'Video pause V01-012-00, id: 012',
			   5: 'Video play V00-098-00, id: 098',
			   6: 'Video pause V01-098-00, id: 098',
			   7: 'Video play V00-019-01, id: 019',
			   8: 'Video pause V01-019-01, id: 019',
			   9: 'Video play V00-005-01, id: 005',
			   10: 'Video pause V01-005-01, id: 005' ,
			   11: 'Video play V00-098-01, id: 098',
			   12: 'Video pause V01-098-01, id: 098',
			   13: 'Video play V01-019-02, id: 019',
			   14: 'Video pause V01-019-02, id: 019',
			   15: 'Video play V01-012-02, id: 012',
			   16: 'Video pause V01-012-02, id: 012',
			   17: 'Video play V01-098-02, id: 098',
			   18: 'Video pause V01-098-02, id: 098'}},
		   5: {'name': 'Makeup',
		   'pai': {0: 'Unknown, id: 000',
			   1: 'Makeup , id: 985',
			   2: 'Makeup , id: 986',
			   3: 'Makeup , id: 987',
			   4: 'Makeup , id: 988',
			   5: 'Makeup , id: 989',
			   6: 'Makeup , id: 990',
			   7: 'Makeup , id: 991',
			   8: 'Makeup , id: 992',
			   9: 'Makeup , id: 993',
			   10: 'Makeup , id: 994',
			   11: 'Makeup , id: 995',
			   12: 'Makeup , id: 996',
			   13: 'Makeup , id: 997',
			   14: 'Makeup , id: 998',
			   15: 'Makeup , id: 999',
			   16: 'Makeup , id: 984',
			   17: 'Makeup , id: 983',
			   18: 'Makeup , id: 982',
			   19: 'Makeup , id: 981',
			   20: 'Makeup , id: 980',
			   21: 'Makeup , id: 979',
			   22: 'Makeup , id: 978',
			   23: 'Makeup , id: 977',
			   24: 'Makeup , id: 976',
			   25: 'Makeup , id: 975',
			   26: 'Makeup , id: 974',
			   27: 'Makeup , id: 973',
			   28: 'Makeup , id: 972',
			   29: 'Makeup , id: 971',
			   30: 'Makeup , id: 970',
			   31: 'Makeup , id: 969',
			   32: 'Makeup , id: 968',
			   33: 'Makeup , id: 967',
			   34: 'Makeup , id: 966',
			   35: 'Makeup , id: 965',
			   36: 'Makeup , id: 964',
			   37: 'Makeup , id: 963',
			   38: 'Makeup , id: 962',
			   39: 'Makeup , id: 961',
			   40: 'Makeup , id: 960',
			   41: 'Makeup , id: 959',
			   42: 'Makeup , id: 958',
			   43: 'Makeup , id: 957',
			   44: 'Makeup , id: 956',
			   45: 'Makeup , id: 955',
			   46: 'Makeup , id: 954',
			   47: 'Makeup , id: 953',
			   48: 'Makeup , id: 952',
			   49: 'Makeup , id: 951',
			   50: 'Makeup , id: 950',
			   51: 'Makeup , id: 949',
			   52: 'Makeup , id: 948',
			   53: 'Makeup , id: 947',
			   54: 'Makeup , id: 946',
			   55: 'Makeup , id: 945',
			   56: 'Makeup , id: 944',
			   57: 'Makeup , id: 943',
			   58: 'Makeup , id: 942',
			   59: 'Makeup , id: 941',
			   60: 'Makeup , id: 940',
			   61: 'Makeup , id: 939',
			   62: 'Makeup , id: 938',
			   63: 'Makeup , id: 937',
			   64: 'Makeup , id: 936',
			   65: 'Makeup , id: 935',
			   66: 'Makeup , id: 934',
			   67: 'Makeup , id: 933',
			   68: 'Makeup , id: 932',
			   69: 'Makeup , id: 931',
			   70: 'Makeup , id: 930',
			   71: 'Makeup , id: 929',
			   72: 'Makeup , id: 928',
			   73: 'Makeup , id: 927',
			   74: 'Makeup , id: 926',
			   75: 'Makeup , id: 925',
			   76: 'Makeup , id: 924',
			   77: 'Makeup , id: 923',
			   78: 'Makeup , id: 922',
			   79: 'Makeup , id: 921',
			   80: 'Makeup , id: 920',
			   81: 'Makeup , id: 919',
			   82: 'Makeup , id: 918',
			   83: 'Makeup , id: 917',
			   84: 'Makeup , id: 916',
			   85: 'Makeup , id: 915',
			   86: 'Makeup , id: 914',
			   87: 'Makeup , id: 913',
			   88: 'Makeup , id: 912',
			   89: 'Makeup , id: 911',
			   90: 'Makeup , id: 910',
			   91: 'Makeup , id: 909',
			   92: 'Makeup , id: 908',
			   93: 'Makeup , id: 907',
			   94: 'Makeup , id: 906',
			   95: 'Makeup , id: 905',
			   96: 'Makeup , id: 904',
			   97: 'Makeup , id: 903',
			   98: 'Makeup , id: 902',
			   99: 'Makeup , id: 901',
			   100: 'Makeup , id: 900',
			   101: 'Makeup , id: 899',
			   102: 'Makeup , id: 898',
			   103: 'Makeup , id: 897',
			   104: 'Makeup , id: 896',
			   105: 'Makeup , id: 895',
			   106: 'Makeup , id: 894',
			   107: 'Makeup , id: 893',
			   108: 'Makeup , id: 892',
			   109: 'Makeup , id: 891',
			   110: 'Makeup , id: 890',
			   111: 'Makeup , id: 889',
			   112: 'Makeup , id: 888',
			   113: 'Makeup , id: 887',
			   114: 'Makeup , id: 886',
			   115: 'Makeup , id: 885',
			   116: 'Makeup , id: 884',
			   117: 'Makeup , id: 883',
			   118: 'Makeup , id: 882',
			   119: 'Makeup , id: 881',
			   120: 'Makeup , id: 880',
			   121: 'Makeup , id: 879',
			   122: 'Makeup , id: 878',
			   123: 'Makeup , id: 877',
			   124: 'Makeup , id: 876',
			   125: 'Makeup , id: 875',
			   126: 'Makeup , id: 874',
			   127: 'Makeup , id: 873',
			   128: 'Makeup , id: 872',
			   129: 'Makeup , id: 871',
			   130: 'Makeup , id: 870',
			   131: 'Makeup , id: 869',
			   132: 'Makeup , id: 868',
			   133: 'Makeup , id: 867',
			   134: 'Makeup , id: 866',
			   135: 'Makeup , id: 865',
			   136: 'Makeup , id: 864',
			   137: 'Makeup , id: 863',
			   138: 'Makeup , id: 862',
			   139: 'Makeup , id: 861'}}}

videos = '/data/zming/datasets/Anti-spoof/WMCA/WMCA-Image/R_CDIT'
savedir = '/data/zming/datasets/Anti-spoof/WMCA/WMCA-Image/protocol/unseen'
#file_name = 'train.csv'
# dataset_file = '/data/zming/datasets/Anti-spoof/WMCA/WMCA-Image/WMCA_Grandtest_revise.txt'
dataset_file = '/data/zming/datasets/Anti-spoof/WMCA/WMCA-Image/protocol/unseen/rigidmask/train_color.csv'
#dataset_file_revise = '/data/zming/datasets/Anti-spoof/WMCA/WMCA-Image/WMCA_Grandtest_revise.txt'

#attacks = ['glasses', 'Mannequin', 'Printed',  'Video', 'rigidmask',  'Flexible', 'Papercraft' ]
attacks = ['rigidmask']
# videos = '/data/zming/datasets/Anti-spoof/CASIA/test_release'
# file_name = 'test.csv'

def main():
	vids = os.listdir(videos)
	attacks_train = np.zeros(7)
	attacks_dev = np.zeros(7)
	attacks_eval = np.zeros(7)
	attacks_train_vid = []
	attacks_dev_vid = []
	attacks_eval_vid = []

	df_video=pd.read_csv(dataset_file, sep=',', index_col=None, header=None,names=['image_path', 'bbox', 'spoofing_label', 'type'])
	image_paths=df_video['image_path']

	for image_path in image_paths:
		vid_name =str.split(image_path, '/')[-3]
		### extract type_id, pai_id in vid_name
		strtmp=str.split(vid_name, '_')
		type_id=int(strtmp[-2])
		pai_id=int(strtmp[-1])

		item=BATL_CONFIG[type_id]['pai'][pai_id]
		if 'Plastic' in item or 'transparent' in item or 'Custom' in item:
			attacks_train[0]+=1
			attacks_train_vid.append(vid_name)
		attacks_train_vid = list(set(attacks_train_vid))

	with open(dataset_file, 'r') as data_file:
		lines=data_file.readlines()
		for line in lines[1:]:
			num, client_id, vid_name, group, label=str.split(line, ',')
			spooflabel='0' if label[:-1] == "attack" else '1'
			vid_name=str.split(vid_name, '/')[-1]
			vid_path=os.path.join(videos, vid_name)
			modalities=os.listdir(vid_path)
			modalities.sort()

			### extract type_id, pai_id in vid_name
			strtmp=str.split(vid_name, '/')[-1]
			strtmp=str.split(strtmp, '_')
			type_id=int(strtmp[-2])
			pai_id=int(strtmp[-1])

			for i, attack in enumerate(attacks):
				if attack == 'rigidmask':
					item=BATL_CONFIG[type_id]['pai'][pai_id]
					if 'Plastic' in item or 'transparent' in item or 'Custom' in item:
						if group == 'train':
							attacks_train[i]+=1
							attacks_train_vid.append(vid_name)
						elif group == 'dev':
							attacks_dev[i]+=1
							attacks_dev_vid.append(vid_name)
						elif group == 'eval':
							attacks_eval[i]+=1
							attacks_eval_vid.append(vid_name)
						else:
							raise valueError("Wrong group! %s" % vid_name)
				elif  attack == 'glasses': ## only glasses in Facial Disguise type_id==1
					item=BATL_CONFIG[type_id]['pai'][pai_id]
					if attack in item and type_id==1:
						if group == 'train':
							attacks_train[i]+=1
							attacks_train_vid.append(vid_name)
						elif group == 'dev':
							attacks_dev[i]+=1
							attacks_dev_vid.append(vid_name)
						elif group == 'eval':
							attacks_eval[i]+=1
							attacks_eval_vid.append(vid_name)
						else:
							raise valueError("Wrong group! %s" % vid_name)
				elif  attack == 'Video': ## including Electronic photo
					if type_id==4 or (type_id==3 and pai_id in [5,10,15]):
						if group == 'train':
							attacks_train[i]+=1
							attacks_train_vid.append(vid_name)
						elif group == 'dev':
							attacks_dev[i]+=1
							attacks_dev_vid.append(vid_name)
						elif group == 'eval':
							attacks_eval[i]+=1
							attacks_eval_vid.append(vid_name)
						else:
							raise valueError("Wrong group! %s" % vid_name)
				else:
					item=BATL_CONFIG[type_id]['pai'][pai_id]
					if attack in item:

						if group == 'train':
							attacks_train[i]+=1
							attacks_train_vid.append(vid_name)
						elif group == 'dev':
							attacks_dev[i]+=1
							attacks_dev_vid.append(vid_name)
						elif group == 'eval':
							attacks_eval[i]+=1
							attacks_eval_vid.append(vid_name)
						else:
							raise valueError("Wrong group! %s" % vid_name)

	# with open(dataset_file_revise, 'w') as data_file_revise:
	# 	with open(dataset_file, 'r') as data_file:
	# 		lines=data_file.readlines()
	# 		data_file_revise.write("%s" % lines[0])
	# 		for line in lines[1:]:
	# 			num, client_id, vid_name_all, group, label=str.split(line, ',')
	# 			vid_name=str.split(vid_name_all, '/')[-1]
	# 			if vid_name in attacks_eval_vid:
	# 				if client_id in ["1", "100"]:
	# 					group='train'
	# 				elif client_id == "12":
	# 					group="dev"
	# 				elif client_id == "99":
	# 					group='eval'
	# 				else:
	# 					raise valueError("wrong client_id %s"%vid_name)
	# 			line = num+","+client_id+","+ vid_name_all+","+ group+","+ label
	# 			data_file_revise.write("%s"%line)

	for attack in attacks:
		videos_select = []
		attack_dir = os.path.join(savedir, attack)
		if not os.path.isdir(attack_dir):
			os.mkdir(attack_dir)
		with open(dataset_file, 'r') as data_file:
			lines = data_file.readlines()
			bbox = '0 0 128 128'
			for line in lines[1:]:
				num,client_id, vid_name, group, label = str.split(line, ',')
				spooflabel = '0' if label[:-1]=="attack" else '1'
				vid_name = str.split(vid_name, '/')[-1]
				vid_path = os.path.join(videos, vid_name)
				modalities = os.listdir(vid_path)
				modalities.sort()

				### extract type_id, pai_id in vid_name
				strtmp = str.split(vid_name, '/')[-1]
				strtmp = str.split(strtmp, '_')
				type_id = int(strtmp[-2])
				pai_id = int(strtmp[-1])
				# if type_id == 2 and pai_id in [8,9,10,11] and group=='train':
				# 	print(vid_name)
				# if vid_name == '001_04_010_3_03':
				# 	print(line)
				if group == 'dev' and type_id != 0:
					videos_select.append(vid_path)

				isSave=False
				if group=='train' or group=='dev':
					if type_id != 0:
						if attack == 'rigidmask':
							item=BATL_CONFIG[type_id]['pai'][pai_id]
							if item == 'Custom wearable mask B05-02-00, id: 002':
								print(i)
							if 'Plastic' in item or 'transparent' in item or 'Custom' in item:
								continue
						elif attack == 'glasses':  ## only glasses in Facial Disguise type_id==1
							item=BATL_CONFIG[type_id]['pai'][pai_id]
							if attack in item and type_id == 1:
								continue
						elif attack == 'Video':  ## including Electronic photo
							if type_id == 4 or (type_id == 3 and pai_id in [5, 10, 15]):
								continue
						else:
							item=BATL_CONFIG[type_id]['pai'][pai_id]
							if attack in item:
								continue
					isSave=True

				if group=='eval':
					if type_id != 0:
						# item = BATL_CONFIG[type_id]['pai'][pai_id] ## attack Printed
						# if attack not in item:
						# 	continue
						#
						# if attack == 'rigidmask':
						# 	if 'Plastic' not in item and 'transparent' not in item and 'Custom' not in item:
						# 		continue
						#
						# # if attack == 'Photo': ## error this will include Electronic photo in eval
						# # 	if type_id != 3:
						# # 		continue
						#
						# if attack != 'rigidmask' and attack != 'Photo' and attack not in item:
						# 	continue


						if attack == 'rigidmask':
							item=BATL_CONFIG[type_id]['pai'][pai_id]
							if 'Plastic' in item or 'transparent' in item or 'Custom' in item:
								isSave = True
						elif attack == 'glasses':  ## only glasses in Facial Disguise type_id==1
							item=BATL_CONFIG[type_id]['pai'][pai_id]
							if attack in item and type_id == 1:
								isSave = True
						elif attack == 'Video':  ## including Electronic photo
							if type_id == 4 or (type_id == 3 and pai_id in [5, 10, 15]):
								isSave = True
						else:
							item=BATL_CONFIG[type_id]['pai'][pai_id]
							if attack in item:
								isSave = True
					else:
						isSave = True

				#for modality in modalities:
				if isSave:
					for modality in modalities:
						print("%s"%os.path.join(vid_path, modality))
						images = glob.glob(os.path.join(vid_path, modality, '*.jpg'))
						images.sort()
						data=[]
						for image in images:
							data+=[[image, bbox, spooflabel]]
						with open(os.path.join(attack_dir, "%s_%s.csv" % (group, modality)), 'a') as cvsfile:
							writer=csv.writer(cvsfile)
							writer.writerows(data)

		print("attack")

if __name__ == '__main__':
	main()

