import librosa
import numpy as np
import torch
import constants
import torchaudio

import torchaudio.transforms as T


class AudioUtils(object):

    @staticmethod
    def read_audio(path, offset, duration):
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
