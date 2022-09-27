import os
import glob
import matplotlib.pyplot as plt
import numpy as np

#log_path = "/data/zming/logs/Antispoof/test/Transfomer_FAS/CASIA/REPLAY/8layers_2heads_length10_sttn_transformer_Efficientnet_pretrainedCASIA_generator/1-1976_intervid1_interframe10"
#log_path = "/data/zming/logs/Antispoof/test/Transfomer_FAS/CASIA/REPLAY/8layers_2heads_length10_sttn_transformer_Efficientnet_pretrainedCASIA_GAN_MTL/1-1586_intervid1_interframe10"
#log_path = "/data/zming/logs/Antispoof/test/tmp"
#log_path = "/data/zming/logs_yau/Antispoof/test/Transformer_FAS/OCIM/CIM-O/8layers_2heads_length10_sttn_transformer_Efficientnet"
#log_path = "/data/zming/logs/Antispoof/test/tmp1"
log_path = "/home/ming_01/DATA/models/Depth_antispoof"
tmp_folders = ["tmp180"]
model= "sttn_transformer_OCIM"
colortype = ['-bx', '-gx','-mx','-yx','-cx']
files_min = 100000


# rename evaluation files name
for tmp_folder in tmp_folders:
    files = glob.glob(os.path.join(log_path,tmp_folder, model, "evaluation_ep*.txt"))
    for file in files:
        num_ep=int(str.split(file, '/')[-1][13:-4])
        os.rename(file, os.path.join(log_path,tmp_folder, model, "evaluation_ep%04d.txt"%num_ep))

## read lr_epochs
lr_file = os.path.join(log_path,tmp_folders[0], model, "log_train.txt")
lr_epochs = []
with open(lr_file, 'r') as f1:
    lines=f1.readlines()
    batch_size_str = str.split(lines[0])[4]
    batch_size = int(str.split(batch_size_str, '/')[-1])
    for line in lines[::5*batch_size]:
        lr=str.split(line, ' ')
        lr=float(lr[-1][:-1])
        lr_epochs.append(lr)

lr_per5epochs = lr_epochs[5::5]
lr_per5epochs_plot =np.array(lr_per5epochs)*1e3


## read the same number of files in the different folders
for tmp_folder in tmp_folders:
    files = glob.glob(os.path.join(log_path,tmp_folder, model, "evaluation_ep*.txt"))
    files_num = len(files)
    if files_num < files_min:
        files_min = files_num


## plot acc, acer, apcer, bpcer, acer of epochs
plt.figure()
fig0=plt.gca()
plt.figure()
fig1=plt.gca()
plt.figure()
fig2=plt.gca()
plt.figure()
fig3=plt.gca()
for i, tmp_folder in enumerate(tmp_folders):
    files = glob.glob(os.path.join(log_path,tmp_folder, model, "evaluation_ep*.txt"))
    files.sort()
    ep_list=[]
    ACC=[]
    BPCER=[]
    APCER=[]
    ACER=[]

    for file in files[:files_min]:
        num_ep = int(str.split(file,'/')[-1][13:-4])
        ep_list.append(num_ep)
        with open(file, 'r') as f1:
            lines = f1.readlines()

            acc, apcer, bpcer, acer = str.split(lines[0], ',')
            acc = float(acc[-6:])
            apcer = float(apcer[-6:])
            bpcer = float(bpcer[-6:])
            acer = float(acer[6:12])

            ACC.append(acc)
            APCER.append(apcer)
            BPCER.append(bpcer)
            ACER.append(acer)

    print(ep_list)


    #plt.plot(ep_list, ACER, colortype[i])
    plt.sca(fig0)
    plt.plot(ACER, colortype[i])
    plt.sca(fig1)
    plt.plot(APCER, colortype[i])
    plt.sca(fig2)
    plt.plot(BPCER, colortype[i])
    plt.sca(fig3)
    plt.plot(ACC, colortype[i])



#plt.figure()
plt.sca(fig0)
plt.plot(lr_per5epochs_plot[:files_min], '-rx')
legend_list = []
for tmp_folder in tmp_folders:
    legend_list.append('ACER %s'%tmp_folder)
legend_list.append('lr')
plt.legend(legend_list)
plt.grid()
plt.savefig(os.path.join(log_path, tmp_folders[0], model, "plot_ACER.jpg"))
#plt.show()

plt.sca(fig1)
plt.plot(lr_per5epochs_plot[:files_min], '-rx')
legend_list = []
for tmp_folder in tmp_folders:
    legend_list.append('APCER %s'%tmp_folder)
legend_list.append('lr')
plt.legend(legend_list)
plt.grid()
plt.savefig(os.path.join(log_path, tmp_folders[0], model, "plot_APCER.jpg"))
#plt.show()

plt.sca(fig2)
plt.plot(lr_per5epochs_plot[:files_min], '-rx')
legend_list = []
for tmp_folder in tmp_folders:
    legend_list.append('BPCER %s'%tmp_folder)
legend_list.append('lr')
plt.legend(legend_list)
plt.grid()
plt.savefig(os.path.join(log_path, tmp_folders[0], model, "plot_BPCER.jpg"))
#plt.show()

plt.sca(fig3)
plt.plot(lr_per5epochs_plot[:files_min], '-rx')
legend_list = []
for tmp_folder in tmp_folders:
    legend_list.append('ACC %s'%tmp_folder)
legend_list.append('lr')
plt.legend(legend_list)
plt.grid()
plt.savefig(os.path.join(log_path, tmp_folders[0], model, "plot_ACC.jpg"))
#plt.show()
