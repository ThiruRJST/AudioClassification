import pandas as pd
import torch
from torch.utils.data import Dataset,DataLoader
import torchaudio
from utils.AudioUtils import AudioUtils, MixUp,OneOf,TimeMasking,FrequencyMasking
import numpy as np
import torchaudio.transforms as T

print(torch.__version__)
print(torchaudio.__version__)


df = pd.read_csv('final.csv')
a = AudioUtils()
#augs = AudioAugmentations()


class AudioDataset(torch.utils.data.Dataset):
    
    def __init__(self,val_fold,mode='train',mixup=False):
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
        self.mixup_bool = mixup
        
                
    def __len__(self):

        if self.mode == 'train':
            return len(self.train)
        else:
            return len(self.val)

    def __getitem__(self, index):

        df = self.train if self.mode == 'train' else self.val

        finame = df.loc[index,'filename']
        foname = df.loc[index,'foldername']
        label = df.loc[index,'class']
        #print(label.size())

        offset = df.loc[index,'start_time']
        dur = df.loc[index,'duration']

        if self.mixup_bool:
            j = index+1
            fname,foname2 = df.loc[j,'filename'],df.loc[j,'foldername']
            label2 = df.loc[j,'class']
            offset2,dur2 = df.loc[j,'start_time'],df.loc[j,'duration']
            audio1 = a.read_audio(filename=finame,foldername=foname,offset=int(offset),duration=int(dur))
            audio2 = a.read_audio(filename=fname,foldername=foname2,offset=int(offset2),duration=int(dur2))

            audio,label = MixUp().apply(audio1,label,audio2,label2)
        else:
            audio = a.read_audio(filename=finame,foldername=foname,offset=int(offset),duration=int(dur))
        audio = a.rechannel(audio)
        audio = a.resample(audio)
        audio = a.pad_audio(audio)

        audio = a.melspectro(audio)

        if self.mode == 'train':
            audio = self.mel_transforms(audio)

        #print(audio.size())

        return audio,label

