import json
import os
import glob

data = "/data/zming/datasets/Youtube-VOS/train/JPEGImages"
output = "/data/zming/datasets/Youtube-VOS/"
vids = os.listdir(data)

img_dict = {}

for vid in vids:
    images=os.listdir(os.path.join(data, vid))
    imgname = int(images[0][:-4])
    img_dict[vid]=imgname

with open(os.path.join(output, 'train.json'), 'w') as outfile:
    json.dump(img_dict, outfile)

