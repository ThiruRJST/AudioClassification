import pandas as pd
import torch
from torch.utils.data import Dataset,DataLoader
import torchaudio
from AudioUtils import AudioUtils,OneOf,TimeMasking,FrequencyMasking
import numpy as np
import torchaudio.transforms as T

print(torch.__version__)
print(torchaudio.__version__)


df = pd.read_csv('160.csv')
a = AudioUtils()
#augs = AudioAugmentations()


class AudioDataset(torch.utils.data.Dataset):
    
    def __init__(self):
        super(AudioDataset, self).__init__()
        
        self.df = df

        self.mel_transforms = OneOf([
            TimeMasking(),
            FrequencyMasking()
        ])

        
                
    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):

        path = self.df.loc[index,'path']+'.wav'
        label = torch.tensor([int(self.df.loc[index,'class'])])
        #print(label.size())

        offset = self.df.loc[index,'start_time']
        dur = self.df.loc[index,'duration']

        audio = a.read_audio(path,int(offset),int(dur))
        audio = a.rechannel(audio)
        audio = a.resample(audio)
        audio = a.pad_audio(audio)

        audio = a.melspectro(audio)

        audio_aug = self.mel_transforms(audio)


            


        #print(audio.size())



        return audio_aug,label


dataset = AudioDataset()
dataset = DataLoader(dataset,batch_size=16,shuffle=True)

print(next(iter(dataset)))