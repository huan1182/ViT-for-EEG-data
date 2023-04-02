import numpy as np
import pandas as pd
from PIL import Image
from torch.utils.data import DataLoader, Dataset
import os
import glob
from sklearn.preprocessing import normalize
from scipy.fft import fft2,fft
import os
import torch
import glob
import numpy as np
from torchvision.transforms import functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms



def eeg_normalize(data_mat,num_features=8):
    result=np.zeros(data_mat.shape)
    for i in range(num_features):
        result[:,i]=data_mat[:,i]/np.linalg.norm(data_mat[:,i])
    result=result*255
    return result
def data_concat(file_path,num_file):
    for i in range(num_file):
        data_path=file_path+'/'+str(i+1)+'.csv'
        temp=pd.read_csv(data_path).to_numpy()
        #print(temp.shape)
        if i==0:
            data_mat=temp
        else:
            ##print('reach_here')
            data_mat=np.concatenate((data_mat,temp))
    #data_mat=eeg_normalize(data_mat)
    #data_mat=np.uint8(data_mat)
    print(data_mat.shape)
    return data_mat

def fft_eeg(data):
    p=data.shape[1]
    result=np.zeros(data.shape)
    for i in range(p):
        result[:,i]=fft(data[:,i])
    return result
def data_processor(data,interval,aug,file_path,do_fft=False,stack_3_ch=False):
    n=data.shape[0]
    p=data.shape[1]
    num_k=int(n/interval-2)
    int_aug=int(interval/aug)
    count=0
    for i in range(num_k):
        for j in range(aug):
            temp=data[(i*interval+j*int_aug):(i*interval+j*int_aug+interval),:]
            temp=np.around(temp)
            temp=np.uint8(temp)
            if do_fft:
                temp=fft_eeg(temp)
            if stack_3_ch:
                temp=np.stack((temp,temp,temp),axis=-1)
            path_data=file_path+'/'+str(count)+'.npy'
            np.save(path_data,temp)
            count+=1
    return 1
def data_creator_unit(inpath,outpath,num_file,interval,aug,do_fft=False,stack_3_ch=False):
    data_mat=data_concat(inpath,num_file)
    data_processor(data=data_mat,interval=interval,aug=aug,file_path=outpath,do_fft=do_fft,stack_3_ch=stack_3_ch)
    return 1

def data_creator(folder_name_in,folder_name_out,num_file,interval,aug,do_fft=False,stack_3_ch=False):
    inpath_train_p=folder_name_in+'/train/positive'
    inpath_train_n=folder_name_in+'/train/negative'
    outpath_train_p=folder_name_out+'/train/positive'
    outpath_train_n=folder_name_out+'/train/negative'
    inpath_test_p=folder_name_in+'/test/positive'
    inpath_test_n=folder_name_in+'/test/negative'
    outpath_test_p=folder_name_out+'/test/positive'
    outpath_test_n=folder_name_out+'/test/negative'
    data_creator_unit(inpath_train_p,outpath_train_p,num_file=num_file[0],interval=interval,aug=aug,do_fft=do_fft,stack_3_ch=stack_3_ch)
    data_creator_unit(inpath_train_n,outpath_train_n,num_file=num_file[1],interval=interval,aug=aug,do_fft=do_fft,stack_3_ch=stack_3_ch)
    data_creator_unit(inpath_test_p,outpath_test_p,num_file=num_file[2],interval=interval,aug=aug,do_fft=do_fft,stack_3_ch=stack_3_ch)
    data_creator_unit(inpath_test_n,outpath_test_n,num_file=num_file[3],interval=interval,aug=aug,do_fft=do_fft,stack_3_ch=stack_3_ch)
    return 1

def data_processor2(data,interval,aug,file_path_train,file_path_test,do_fft=False,stack_3_ch=False):
    n=data.shape[0]
    p=data.shape[1]
    num_k=int(n/interval/2-2)
    int_aug=int(interval/aug)
    count=0
    for i in range(num_k):
        for j in range(aug):
            temp=data[(2*i*interval+j*int_aug):(2*i*interval+j*int_aug+interval),:]
            if do_fft:
                temp=fft_eeg(temp)
            if stack_3_ch:
                temp=np.stack((temp,temp,temp),axis=-1)
            path_data=file_path_train+'/'+str(count)+'.npy'
            np.save(path_data,temp)
            temp=data[(2*i*interval+j*int_aug):(2*i*interval+j*int_aug+interval*2),:]
            if do_fft:
                temp=fft_eeg(temp)
            if stack_3_ch:
                temp=np.stack((temp,temp,temp),axis=-1)
            path_data=file_path_test+'/'+str(count)+'.npy'
            np.save(path_data,temp)
            count+=1
    return 1



def data_creator_unit2(inpath,outpath_train,outpath_test,num_file,interval,aug,do_fft=False,stack_3_ch=False):
    data_mat=data_concat(inpath,num_file)
    data_processor2(data=data_mat,interval=interval,aug=aug,file_path_train=outpath_train,file_path_test=outpath_test,do_fft=do_fft,stack_3_ch=stack_3_ch)
    return 1

class Resize(object):
    def __init__(self, size):
        assert isinstance(size, int) or (isinstance(size, collections.Iterable) and len(size) == 2)
        if isinstance(size, int):
            self._size = (size, size)
        else:
            self._size = size
        
    def __call__(self, T):
        return F.resize(T, size=self._size)




def data_creator2(folder_name_in,folder_name_out,num_file,interval,aug,do_fft=False,stack_3_ch=False):
    inpath_p=folder_name_in+'/positive'
    inpath_n=folder_name_in+'/negative'
    outpath_train_p=folder_name_out+'/train/positive'
    outpath_train_n=folder_name_out+'/train/negative'
    outpath_test_p=folder_name_out+'/test/positive'
    outpath_test_n=folder_name_out+'/test/negative'
    data_creator_unit2(inpath_p,outpath_train=outpath_train_p,outpath_test=outpath_test_p,
                       num_file=num_file[0],interval=interval,aug=aug,do_fft=do_fft,stack_3_ch=stack_3_ch)
    data_creator_unit2(inpath_n,outpath_train=outpath_train_n,outpath_test=outpath_test_n,
                       num_file=num_file[1],interval=interval,aug=aug,do_fft=do_fft,stack_3_ch=stack_3_ch)
    return 1







class EEGDataset(Dataset):

    class_to_id = {"positive": 0, "negative": 1}
    
    def __init__(self, data_root, transform=None,DO_FFT=False):
        super().__init__()
        self.data_root = data_root
        self.transform = transform
        self.eeg_files = glob.glob(os.path.join(self.data_root, "**", "*.npy"))
        self.DO_FFT=DO_FFT
        # print(self.eeg_files)
 
    def __getitem__(self, index):
        eeg_file = self.eeg_files[index]
        eeg_data = np.load(eeg_file)
        if self.DO_FFT:
            eeg_data=fft2(eeg_data)
        #eeg_data=normalize(eeg_data)*255
        #eeg_data=np.uint8(eeg_data)
        #eeg_data=Image.fromarray(eeg_data)
        label = os.path.basename(os.path.dirname(eeg_file))

        if self.transform is not None:
            eeg_data = self.transform(eeg_data)
        return eeg_data, self.class_to_id[label]
 
    def __len__(self):
        return len(self.eeg_files)
    


class Resize(object):
    def __init__(self, size):
        assert isinstance(size, int) or (isinstance(size, collections.Iterable) and len(size) == 2)
        if isinstance(size, int):
            self._size = (size, size)
        else:
            self._size = size
        
    def __call__(self, T):
        return F.resize(T, size=self._size)