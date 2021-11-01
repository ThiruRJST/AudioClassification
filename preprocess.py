import pandas as pd
import torch
from torch.utils.data import Dataset,DataLoader
import torchaudio
from utils.AudioUtils import AudioUtils,OneOf,TimeMasking,FrequencyMasking
import numpy as np
import torchaudio.transforms as T

print(torch.__version__)
print(torchaudio.__version__)


df = pd.read_csv('160.csv')
a = AudioUtils()
#augs = AudioAugmentations()


class AudioDataset(torch.utils.data.Dataset):
    
    def __init__(self,val_fold,mode='train'):
        super(AudioDataset, self).__init__()
        self.val_fold = val_fold
        self.df = df
        self.mode = mode
        self.train = self.df[self.df.folds != self.val_fold].reset_index(drop=True)
        self.val = self.df[self.df.folds == self.val_fold].reset_index(drop=True)

        self.mel_transforms = OneOf([
            TimeMasking(),
            FrequencyMasking()
        ])

        
                
    def __len__(self):

        if self.mode == 'train':
            return len(self.train)
        else:
            return len(self.val)

    def __getitem__(self, index):

        df = self.train if self.mode == 'train' else self.val

        path = df.loc[index,'path']+'.wav'
        label = torch.tensor([int(df.loc[index,'class'])])
        #print(label.size())

        offset = df.loc[index,'start_time']
        dur = df.loc[index,'duration']

        audio = a.read_audio(path,int(offset),int(dur))
        audio = a.rechannel(audio)
        audio = a.resample(audio)
        audio = a.pad_audio(audio)

        audio = a.melspectro(audio)

        if self.mode == 'train':
            audio_aug = self.mel_transforms(audio)

        #print(audio.size())

        return audio_aug,label

