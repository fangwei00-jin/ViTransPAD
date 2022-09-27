import json
import pandas as pd
import os
import glob

class csv_protocol_json():
    def __init__(self, data_file, json_file):
        self.data_file = data_file
        self.json_file = json_file
        self.df_writers = pd.read_csv(self.data_file, sep=',', index_col=None, header=None,
                                names=['image_path', 'bbox', 'writer_id'])

    def to_json(self):
        data = {}
        image_paths = self.df_writers['image_path']
        writer_ids = self.df_writers['writer_id']
        bboxes = self.df_writers['bbox']

        writer_id_preced = ''
        for i, image in enumerate(image_paths):
            print(image)
            writer_id = int(writer_ids[i])

            if writer_id != writer_id_preced:
                print(image)
                data[writer_id] = {'spoofing_label': writer_id, 'bboxes': [], 'frames': []}

            bbox = bboxes[i]
            data[writer_id]['bboxes'].append(bbox)
            data[writer_id]['frames'].append(image)

            writer_id_preced = writer_id


        data_json = json.dumps(data) ## should firstly transform to json data before write to a json file
        with open(self.json_file, 'w') as outfile:
            outfile.write(data_json)

if __name__ == '__main__':
    train =  csv_protocol_json('/data/zming/datasets/writer_id/Firemaker/train/train.csv', '/data/zming/datasets/writer_id/Firemaker/train/train.json')
    #train =  csv_protocol_json('/data/zming/datasets/writer_id/Firemaker/test/test.csv', '/data/zming/datasets/writer_id/Firemaker/test/test.json')
    train.to_json()

