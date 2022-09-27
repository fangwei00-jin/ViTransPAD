import os

#log_path = "/data/zming/logs/Antispoof/test/Transfomer_FAS/CASIA/REPLAY/8layers_2heads_length10_sttn_transformer_Efficientnet_pretrainedCASIA_generator/1-1976_intervid1_interframe10"
#log_path = "/data/zming/logs/Antispoof/test/Transfomer_FAS/CASIA/REPLAY/8layers_2heads_length10_sttn_transformer_Efficientnet_pretrainedCASIA_GAN_MTL/1-1586_intervid1_interframe10"
#log_path = "/data/zming/logs/Antispoof/test/tmp"
#log_path = "/data/zming/logs_yau/Antispoof/test/Transformer_FAS/OCIM/CIM-O/8layers_2heads_length10_sttn_transformer_Efficientnet"
#log_path = "/data/zming/logs/Antispoof/test/tmp1"
log_path = "/data/zming/logs/Antispoof/test/tmp2"

ACER=[]
ACER_enc=[]
ACER_dec=[]
ACER_vote=[]
ep_list = []

folders = os.listdir(log_path)
folders.sort()

for folder in folders:
    #ep_folders = os.listdir(os.path.join(log_path, folder))
    ep_folders = os.listdir(os.path.join(log_path))
    ep_folders.sort()
    for ep in ep_folders:
        #ep_path = os.path.join(log_path, folder, ep)
        ep_path = os.path.join(log_path, ep)
        if os.path.isdir(ep_path):
            if os.path.isdir(ep_path):
                num_ep = int(ep[3:])
                # os.renames(ep_path, os.path.join(log_path,folder, "ep_%04d"%num_ep))

                log_file = os.path.join(ep_path, 'log.txt')
                if os.path.isfile(log_file):
                    with open(log_file, 'r') as f1:
                        lines = f1.readlines()

                        acer = str.split(lines[0], ',')[-1]
                        acer = float(acer[7:-1])
                        ACER.append(acer)
                        ep_list.append(num_ep)
# ACER=[]
# ACER_enc=[]
# ACER_dec=[]
# ACER_vote=[]
# for ep in ep_folders:
#     if os.path.isdir(os.path.join(log_path, ep)):
#         log_file = os.path.join(log_path,ep,'log.txt')
#         with open(log_file, 'r') as f1:
#             lines = f1.readlines()
#
#             acer = str.split(lines[0],',')[-1]
#             acer = float(acer[7:-1])
#             ACER.append(acer)
#
#             # acer = str.split(lines[2],',')[-1]
#             # acer = float(acer[11:-1])
#             # ACER_enc.append(acer)
#             #
#             # acer = str.split(lines[4],',')[-1]
#             # acer = float(acer[11:-1])
#             # ACER_dec.append(acer)
#             #
#             # acer = str.split(lines[6],',')[-1]
#             # acer = float(acer[11:-1])
#             # ACER_vote.append(acer)

with open(os.path.join(log_path, 'ACER_all_0.txt'), 'w') as f:
    f.write("ACER_all\n")
    for acer in ACER:
        f.write("%f\n"%acer)
with open(os.path.join(log_path, 'ep_list.txt'), 'w') as f:
    f.write("epoch\n")
    for ep in ep_list:
        f.write("%d\n"%ep)
# with open(os.path.join(log_path, 'ACER_enc_all_0.txt'), 'w') as f:
#     f.write("ACER_enc_all\n")
#     for acer in ACER_enc:
#         f.write("%f\n"%acer)
# with open(os.path.join(log_path, 'ACER_dec_all_0.txt'), 'w') as f:
#     f.write("ACER_dec_all\n")
#     for acer in ACER_dec:
#         f.write("%f\n"%acer)
# with open(os.path.join(log_path, 'ACER_vote_all_0.txt'), 'w') as f:
#     f.write("ACER_vote_all\n")
#     for acer in ACER_vote:
#         f.write("%f\n"%acer)




