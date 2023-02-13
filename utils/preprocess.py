import scipy
from scipy.signal import stft
import numpy as np
import torch.nn.functional as F
import torch
import scipy.io.wavfile as wav

def csv2input(csv_file, start=0, len=1024):
    data = np.loadtxt(csv_file)[start:start+len]
    nperseg = int(np.sqrt(len)) * 2
    downsample = int(nperseg // 64)
    input = data_transofrm(data, nperseg=nperseg, downsample=downsample)
    # print(input.shape)
    return input

def wav2input(wav_file, start=0, len=1024):
    data = np.array(wav.read(wav_file)[1])
    data = data[start:start+len, 0]
    nperseg = int(np.sqrt(len)) * 2
    downsample = int(nperseg // 64)
    input = data_transofrm(data, nperseg=nperseg, downsample=downsample)
    # print(input.shape)
    return input

def IQwav2input(wav_file, start=0, len=1024):
    data = np.array(wav.read(wav_file)[1])
    data = data[start:start+len, 0] + data[start:start+len, 1] * 1j
    
    nperseg = int(np.sqrt(len)) * 2
    downsample = int(nperseg // 64)
    input = data_transofrm(data, nperseg=nperseg, downsample=downsample)
    # print(input.shape)
    return input    

def IQwav2input1d(wav_file, start=0, len=1024):
    data = np.array(wav.read(wav_file)[1])
    print(data.shape)
    # data = (data/data.max(0)).transpose((1,0))
    data = data[start:start+len]
    data /= data.max(0)
    data = data.transpose((1,0))
    # data = data.transpose((1,0))
    return torch.tensor(data).reshape(1, 2, len)

def data_transofrm(x, nperseg=64, downsample=1):
    _, _, Z = stft(x, nperseg=nperseg, return_onesided=False)
    ih, iw = downsample * 32, downsample * 32
    E = torch.tensor(np.abs(Z[len(Z)//2-1:-1, :-1])).reshape(1, 1, ih, iw) # 裁剪，求模值
    E = F.interpolate(E, (32, 32))  # 下采样
    E = torch.log10(E).float() # 取对数
    return E