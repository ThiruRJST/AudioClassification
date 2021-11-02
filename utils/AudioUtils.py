import os
import librosa
import numpy as np
import torch
import constants
import torchaudio

import torchaudio.transforms as T


class AudioUtils(object):

    @staticmethod
    def read_audio(filename,foldername, offset, duration):
        path = os.path.join(foldername,filename)
        audio, sr = torchaudio.load(
            path, frame_offset=offset*constants.SAMPLING_RATE, num_frames=duration*constants.SAMPLING_RATE)
        return (audio, sr)

    @staticmethod
    def rechannel(aud, channels=constants.NUM_CHANNELS):
        sig, sr = aud
        n_ch = sig.shape[0]

        if n_ch == channels:
            return aud

        else:
            rechanneled = torch.cat([sig, sig])

        return (rechanneled, sr)

    @staticmethod
    def resample(aud, new_samp_rate=constants.SAMPLING_RATE):

        sig, sr = aud

        if (sr == new_samp_rate):
            return aud
        else:
            resampled = T.Resample(sr, new_samp_rate)(sig)

        return (resampled, sr)

    @staticmethod
    def pad_audio(aud, max_len=constants.DURATION):
        sig, sr = aud
        rows, sig_len = sig.shape
        max_len = max_len * constants.SAMPLING_RATE

        if (sig_len == max_len):
            return aud

        elif (sig_len > max_len):
            sig_len = sig_len[:, :max_len]

        elif (sig_len < max_len):

            pad = (max_len - sig_len) // 2

            pad_begin = torch.zeros((rows, pad))
            pad_end = torch.zeros((rows, pad))

            padded = torch.cat((pad_begin, sig, pad_end),1)

        return (padded, sr)

    @staticmethod
    def melspectro(aud, n_mels=constants.N_MELS, n_fft=constants.N_FFT, hop_len=constants.HOP_LENGTH):

        sig, sr = aud

        melspec = T.MelSpectrogram(
            sr, n_mels=n_mels, n_fft=n_fft, hop_length=hop_len)(sig)

        melspec = T.AmplitudeToDB(top_db=80)(melspec)

        return melspec


class OneOf:
    '''
    Selects any one of the given transforms in the pipeline
    '''
    def __init__(self,transforms:list):
        self.transforms = transforms
    
    def __call__(self,mel:np.ndarray):
        n_transforms = len(self.transforms)
        trns_index = np.random.choice(n_transforms)
        trns = self.transforms[trns_index]
        return trns(mel)



class AudioAugmentations:
    def __init__(self,always_apply = True,p=0.5):
        self.always_apply = always_apply
        self.p = p

    def __call__(self,mel:torch.Tensor):
        if self.always_apply:
            return self.apply(mel)
        else:
            if np.random.rand() < self.p:
                return self.apply(mel)
            
            else:
                return mel
        
        
    
    

class TimeMasking(AudioAugmentations):

    def __init__(self, always_apply=True, p=0.5,time_mask_param = constants.TIME_MASK_PARAM):
        super().__init__(always_apply=always_apply, p=p)
        self.time_mask_param = time_mask_param


    def apply(self,mel:torch.Tensor):
        time_mask = T.TimeMasking(self.time_mask_param)
        return time_mask(mel)

class FrequencyMasking(AudioAugmentations):

    def __init__(self, always_apply=True, p=0.5,freq_mask_param = constants.FREQ_MASK_PARAM):
        super().__init__(always_apply=always_apply, p=p)
        self.freq_mask_param = freq_mask_param
    
    def apply(self,mel:torch.Tensor):
        freq_mask = T.FrequencyMasking(self.freq_mask_param)
        return freq_mask(mel)


class MixUp(AudioAugmentations):

    def __init__(self, always_apply=True, p=0.5):
        super().__init__(always_apply=always_apply, p=p)
    
    def apply(self,aud1:torch.Tensor,label1,aud2:torch.Tensor,label2):
        alpha = np.random.uniform(0,1)
        b_alpha = 1 - alpha
        x = torch.add(torch.mul(aud1[0],alpha),torch.mul(aud2[0],b_alpha))
        y = (alpha * label1) + (b_alpha * label2)

        return x,y



    

    
