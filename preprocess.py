import pandas as pd
import torch
from torch.data.utils import Dataset,DataLoader
import torchaudio
import torchaudio.transforms as T

print(torch.__version__)
print(torchaudio.__version__)


df = pd.read_csv('160.csv')


class AudioDataset(torch.utils.data.Dataset):
    """Some Information about MyDataset"""
    def __init__(self):
        super(AudioDataset, self).__init__()
        self.df = pd.read_csv()
        


    def __getitem__(self, index):
        return 

    def __len__(self):
        return 