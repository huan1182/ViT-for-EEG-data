import numpy as np
import pandas as pd
from PIL import Image
from torch.utils.data import DataLoader, Dataset
import os
import glob
def data_concat(file_path,num_file):
    for i in range(num_file):
        data_path=file_path+'/'+str(i+1)+'.csv'
        temp=pd.read_csv(data_path).to_numpy()[:,1:9]
        if i==0:
            data_mat=temp
        else:
            data_mat=np.concatenate((data_mat,temp))
    return data_mat
def data_processor(data,interval,aug,file_path):
    n=data.shape[0]

    num_k=int(n/interval-2)
    int_aug=int(interval/aug)
    count=0
    for i in range(num_k):
        for j in range(aug):
            temp=data[(i*interval+j*int_aug):(i*interval+j*int_aug+interval),:]
            path_data=file_path+'/'+str(count)+'.npy'
            np.save(path_data,temp)
            count+=1
    return 1
def data_creator(inpath,outpath,num_file,interval,aug):
    data_mat=data_concat(inpath,num_file)
    data_mat=normalize(data_mat)
    data_processor(data=data_mat,interval=interval,aug=aug,file_path=outpath)
    return 1

def normalize(data):
    coef_shift=np.amax(np.absolute(data))
    temp=data+coef_shift
    coef_normal=np.amax(np.absolute(data))
    temp=temp/coef_normal
    temp=np.uint8(temp*255)
    return temp

class EEGDataset(Dataset):

    class_to_id = {"positive": 0, "negative": 1}
    
    def __init__(self, data_root, transform=None):
        super().__init__()
        self.data_root = data_root
        self.transform = transform
        self.eeg_files = glob.glob(os.path.join(self.data_root, "**", "*.npy"))
        # print(self.eeg_files)
 
    def __getitem__(self, index):
        eeg_file = self.eeg_files[index]
        eeg_data = np.load(eeg_file)
        
        label = os.path.basename(os.path.dirname(eeg_file))

        if self.transform is not None:
            eeg_data = self.transform(eeg_data)

        return eeg_data, self.class_to_id[label]
 
    def __len__(self):
        return len(self.eeg_files)