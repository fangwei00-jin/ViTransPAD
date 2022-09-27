import json
import pandas as pd
import os
import glob

class csv_protocol_json():
    def __init__(self, data_file, json_file, dataset):
        self.data_file = data_file
        self.json_file = json_file
        self.df_video = pd.read_csv(self.data_file, sep=',', index_col=None, header=None,
                                names=['image_path', 'bbox', 'spoofing_label', 'type'])
        self.dataset = dataset

    def to_json(self):
        data = {}
        image_paths = self.df_video['image_path']
        spoofing_labels = self.df_video['spoofing_label']
        bboxes = self.df_video['bbox']

        # videos = [str.split(vid, '/')[8] for vid in image_path]
        # videos = list(set(videos))
        # videos.sort()

        video_preced = ''
        for i, image in enumerate(image_paths):

            if self.dataset == "SiW" or self.dataset == "OuluNPU" :
                ## SiW, OuluNPU, WMCA
                video = str.split(image, '/')[-2]
            elif self.dataset == "CASIA" or self.dataset == "REPLAY":
                ##CASIA, REPLAY
                video = str.split(image, '/')[-3:-1]
                video = video[0]+'_'+video[1]
            elif self.dataset == "WMCA":
                ## WMCA
                video = str.split(image, '/')[-3]
            elif self.dataset == "OCIM":
                if "CASIA" in image or "REPLAY" in image:
                    ## CASIA or REPLAY
                    video = str.split(image, '/')[-3:-1]
                    video = video[0]+'_'+video[1]
                else:
                    ## OuluNPU or MSU
                    video = str.split(image, '/')[-2]

            else:
                print("Input the dataset!")
                return

            print(image)
            if video != video_preced:
                print(image)
                #tmp_dict = data.get(video, {'spoofing_label': 0, 'bboxes': [], 'frames': []})
                data[video] = {'spoofing_label': 0, 'bboxes': [], 'frames': []}
                label = spoofing_labels[i]
                data[video]['spoofing_label'] = int(label) ## int64 cannot be serialized by json

            bbox = bboxes[i]

            data[video]['bboxes'].append(bbox)
            data[video]['frames'].append(image)

            video_preced = video

        strs = str.split(self.data_file, '/')
        strs = strs[:-1]
        savedir  = '/'.join(strs)
        data_json = json.dumps(data) ## should firstly transform to json data before write to a json file
        with open(os.path.join(savedir, self.json_file), 'w') as outfile:
            outfile.write(data_json)

if __name__ == '__main__':
    train =  csv_protocol_json('/data/zming/datasets/Anti-spoof/OCIM/train/file_csv_OM.csv', '/data/zming/datasets/Anti-spoof/OCIM/train/file_csv_OM.json', 'OCIM')
    train.to_json()

    # csv_folder="/data/zming/datasets/Anti-spoof/WMCA/WMCA-Image/protocol/grandtest"
    # csv_files = glob.glob(os.path.join(csv_folder, '*.csv')) ##WMCA unseen
    # #csv_files = glob.glob(os.path.join(csv_folder, '*.csv')) ##WMCA grandtest
    # csv_files.sort()
    # for csv_file in csv_files:
    #     csv_file_name = str.split(csv_file, '/')[-1]
    #     csv_file_name = csv_file_name[:-4]
    #     train =  csv_protocol_json(csv_file, '%s.json'%csv_file_name, "WMCA")
    #     train.to_json()

    # csv_folder = "/data/zming/datasets/Anti-spoof/WMCA/WMCA-Image/protocol/unseen"
    # folders = os.listdir(csv_folder)
    # folders.sort()
    # for folder in folders:
    #     csv_files = glob.glob(os.path.join(csv_folder, folder, '*.csv')) ##WMCA unseen
    #     #csv_files = glob.glob(os.path.join(csv_folder, '*.csv')) ##WMCA grandtest
    #     csv_files.sort()
    #     for csv_file in csv_files:
    #         csv_file_name = str.split(csv_file, '/')[-1]
    #         csv_file_name = csv_file_name[:-4]
    #         train =  csv_protocol_json(csv_file, '%s.json'%csv_file_name, "WMCA")
    #         train.to_json()

    # train =  csv_protocol_json('/data/zming/datasets/Anti-spoof/OCIM/test/OuluNPU_test.csv', 'OuluNPU_test.json')
    # train.to_json()
    # train =  csv_protocol_json('/data/zming/datasets/Anti-spoof/MSU-MFSD/scene01/MSU.csv', 'MSU.json')
    # train.to_json()
    # train =  csv_protocol_json('/data/zming/datasets/Anti-spoof/SELFCOLLECT_Data/AriadNext/Dataset_PAD_L3i_v1_14_04_2021/train/train.csv', 'train.json')
    # train.to_json()
    # test =  csv_protocol_json('/data/zming/datasets/Anti-spoof/SELFCOLLECT_Data/AriadNext/exemple_video_attaque_avec_flash/test.csv', 'test.json')
    # test.to_json()
    # train =  csv_protocol_json('/data/zming/datasets/SiW/SiW_release/Train/train.csv', 'train.json')
    # train.to_json()
    #
    # test =  csv_protocol_json('/data/zming/datasets/SiW/SiW_release/Test/test.csv', 'test.json')
    # test.to_json()

    # train_protocol1 =  csv_protocol_json('/data/zming/datasets/SiW/SiW_release/Train/train_protocol1.csv', 'train_protocol1.json')
    # train_protocol1.to_json()
    #
    # train_protocol2_medium123 =  csv_protocol_json('/data/zming/datasets/SiW/SiW_release/Train/train_protocol2_medium123.csv', 'train_protocol2_medium123.json')
    # train_protocol2_medium123.to_json()
    #
    # train_protocol2_medium124 =  csv_protocol_json('/data/zming/datasets/SiW/SiW_release/Train/train_protocol2_medium124.csv', 'train_protocol2_medium124.json')
    # train_protocol2_medium124.to_json()
    #
    # train_protocol2_medium134 =  csv_protocol_json('/data/zming/datasets/SiW/SiW_release/Train/train_protocol2_medium134.csv', 'train_protocol2_medium134.json')
    # train_protocol2_medium134.to_json()
    #
    # train_protocol2_medium234 =  csv_protocol_json('/data/zming/datasets/SiW/SiW_release/Train/train_protocol2_medium234.csv', 'train_protocol2_medium234.json')
    # train_protocol2_medium234.to_json()
    #
    # train_protocol3_type12 =  csv_protocol_json('/data/zming/datasets/SiW/SiW_release/Train/train_protocol3_type12.csv', 'train_protocol3_type12.json')
    # train_protocol3_type12.to_json()
    #
    # train_protocol3_type13 =  csv_protocol_json('/data/zming/datasets/SiW/SiW_release/Train/train_protocol3_type13.csv', 'train_protocol3_type13.json')
    # train_protocol3_type13.to_json()

    #
    # test_protocol1 =  csv_protocol_json('/data/zming/datasets/SiW/SiW_release/Test/test_protocol1.csv', 'test_protocol1.json')
    # test_protocol1.to_json()
    #
    # test_protocol2_medium123 =  csv_protocol_json('/data/zming/datasets/SiW/SiW_release/Test/test_protocol2_medium1.csv', 'test_protocol2_medium1.json')
    # test_protocol2_medium123.to_json()
    #
    # test_protocol2_medium124 =  csv_protocol_json('/data/zming/datasets/SiW/SiW_release/Test/test_protocol2_medium2.csv', 'test_protocol2_medium2.json')
    # test_protocol2_medium124.to_json()
    #
    # test_protocol2_medium134 =  csv_protocol_json('/data/zming/datasets/SiW/SiW_release/Test/test_protocol2_medium3.csv', 'test_protocol2_medium3.json')
    # test_protocol2_medium134.to_json()
    #
    # test_protocol2_medium234 =  csv_protocol_json('/data/zming/datasets/SiW/SiW_release/Test/test_protocol2_medium4.csv', 'test_protocol2_medium4.json')
    # test_protocol2_medium234.to_json()
    #
    # test_protocol3_type12 =  csv_protocol_json('/data/zming/datasets/SiW/SiW_release/Test/test_protocol3_type12.csv', 'test_protocol3_type12.json')
    # test_protocol3_type12.to_json()
    #
    # test_protocol3_type13 =  csv_protocol_json('/data/zming/datasets/SiW/SiW_release/Test/test_protocol3_type13.csv', 'test_protocol3_type13.json')
    # test_protocol3_type13.to_json()

    #csv2json_train =  csv_protocol_json('/data/zming/datasets/Anti-spoof/OuluNPU/Train_files/train_protocol3_12346.csv', 'train_protocol3_12346.json', 'OuluNPU')
    # csv2json_train = csv_protocol_json('/data/zming/datasets/SiW/SiW_release/Train/train_protocol3_type12.csv',
    #                                    'train_protocol3_type12.json', 'SiW')
    # csv2json_train.to_json()
    # csv2json_test =  csv_protocol_json('/data/zming/datasets/Anti-spoof/OuluNPU/Test_files/test_protocol1.csv', 'test_protocol1.json')
    # csv2json_test.to_json()
    # csv2json_test =  csv_protocol_json('/data/zming/datasets/Anti-spoof/OuluNPU/Dev_files/dev_protocol1.csv', 'dev_protocol1.json')
    # csv2json_test.to_json()


    # # dir = '/data/zming/datasets/Anti-spoof/OuluNPU/Train_files'
    # # dir = '/data/zming/datasets/Anti-spoof/CASIA/train_release'
    # #dir = '/data/zming/datasets/Anti-spoof/REPLAY/train'
    # dir = '/data/zming/datasets/Anti-spoof/SELFCOLLECT_Data/AriadNext/exemple_video_attaque_avec_flash/'
    # csv_files = glob.glob(os.path.join(dir, '*.csv'))
    # csv_files.sort()
    # for csv_file in csv_files:
    #     file_name = str.split(csv_file, '/')[-1]
    #     file_name = file_name[:-4]
    #     json_file = file_name+'.json'
    #     csv2json_obj =  csv_protocol_json(csv_file, json_file)
    #     csv2json_obj.to_json()