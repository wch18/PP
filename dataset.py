import torch
import numpy as np
import random
import h5py
from scipy.signal import stft
from torch.utils.data import DataLoader, Dataset

class RFMCDataset(Dataset):
    def __init__(self, dataset=r'../sub_dataset/', num_classes=24, samples_per_snr=1600, snr_gate = 0, train=True, transform='stft'):
        
        self.train = train
        self.dataset = dataset
        self.transform = transform
        self.num_classes = num_classes
        self.samples_per_snr = samples_per_snr
        self.snr_gate = snr_gate
        self.samples_per_modu = samples_per_snr * (26 - snr_gate)
        self.offset = self.snr_gate * samples_per_snr

    def data_transofrm(self, x, nperseg=64):
        x = x[:,0] + x[:, 1]
        _, _, Z = stft(x, nperseg=nperseg, return_onesided=False)
        Z = Z[len(Z)//2-1:-1, :-1]
        E = np.log10(np.abs(Z)).reshape((1,32,32))
        # E = np.abs(Z)
        E = E / np.std(E, axis=0)
        # E = E.reshape((1,32,32))
        return E
    
    def __getitem__(self, index):
        
        modu = int(index // self.samples_per_modu)
        num = int(self.offset + index % self.samples_per_modu)
        if self.train:
            h5file = self.dataset + 'train/' + 'modu_' + str(modu) + '.h5'
        else:
            h5file = self.dataset + 'val/' + 'modu_' + str(modu) + '.h5'
        h5 = h5py.File(h5file, 'r')
        x = h5['X'][num]
        y = np.concatenate([h5['Y'][num], h5['Z'][num]])
        if self.transform == 'stft':
            x_transform = self.data_transofrm(x)
        else:
            x_transform = (x/x.max(0)).transpose((1, 0))

        return x_transform, y

    def __len__(self):
        return int(self.samples_per_modu * self.num_classes)