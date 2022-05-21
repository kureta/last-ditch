# main.py
import librosa
import numpy as np
import torch
import torchcrepe
from pytorch_lightning.utilities.cli import LightningCLI

from torch.utils.data import Dataset, DataLoader
from pytorch_lightning import LightningDataModule, LightningModule

from dsp import HarmonicOscillator, FilteredNoise, ConvolutionalReverb
from loss import distance
from model import Controller
from soundfile import write as write_wav
from constants import *


class DDSP(LightningModule):
    def __init__(self,
                 n_harmonics: int = 128,
                 n_filters: int = 64,
                 in_ch: int = 1,
                 out_ch: int = 2,
                 reverb_dur: int = 3,
                 lr=0.003,
                 ):
        super().__init__()
        self.controller = Controller(n_harmonics, n_filters)
        self.harmonics = HarmonicOscillator(n_harmonics, in_ch)
        self.noise = FilteredNoise(n_filters, in_ch)
        self.reverb = ConvolutionalReverb(reverb_dur, in_ch, out_ch)
        self.lr = lr

    def forward(self, pitch, loudness):
        harm_ctrl, noise_ctrl = self.controller(pitch, loudness)
        harm = self.harmonics(*harm_ctrl)
        noise = self.noise(noise_ctrl)
        out = self.reverb(harm + noise)

        return out

    def training_step(self, batch, batch_nb):
        f0, amp, x = batch
        y = self(f0, amp)
        loss = distance(x, y)

        if batch_nb % 100 == 0:
            print(y.shape)
            print('==================')
            write_wav('./sick.wav', y[0].detach().cpu().numpy().T, SAMPLE_RATE)

        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)


TEST_FILE_PATH = '/home/kureta/Music/cello-test.wav'


class AudioDataset(Dataset):
    def __init__(self):
        super().__init__()
        audio, sr = librosa.load(TEST_FILE_PATH, sr=SAMPLE_RATE, mono=False)
        mono_audio, sr = librosa.load(TEST_FILE_PATH, sr=SAMPLE_RATE, mono=True)
        self.audio = torch.from_numpy(audio[:, :-(audio.shape[-1] % HOP_LENGTH)]).type(torch.FloatTensor)
        mono_audio = mono_audio[None, :]
        mono_audio = mono_audio[:, :-(mono_audio.shape[-1] % HOP_LENGTH)]

        resampled_audio = librosa.resample(mono_audio, orig_sr=SAMPLE_RATE, target_sr=16000)
        freq, _, _ = torchcrepe.predict(torch.from_numpy(resampled_audio), 16000,
                                        hop_length=HOP_LENGTH // 3,
                                        decoder=torchcrepe.decode.weighted_argmax,
                                        device='cuda', return_periodicity=True)
        self.freq = freq.type(torch.FloatTensor)

        stft = librosa.stft(mono_audio, n_fft=N_FFT, hop_length=HOP_LENGTH)
        freqs = librosa.core.fft_frequencies(sr=SAMPLE_RATE, n_fft=N_FFT)
        weights = librosa.A_weighting(freqs)
        xmag = weights[None, :, None] + librosa.amplitude_to_db(np.abs(stft))
        self.loudness = torch.from_numpy(np.mean(xmag, axis=-2)).type(torch.FloatTensor)

    def __len__(self):
        return 1

    def __getitem__(self, item):
        return self.freq, self.loudness, self.audio


class AudioDataModule(LightningDataModule):
    def __init__(self):
        super().__init__()
        audio, sr = librosa.load(TEST_FILE_PATH, sr=SAMPLE_RATE, mono=False)
        if len(audio.shape) == 1:
            audio = audio[None, :]
        self.audio = audio[:, :-(len(audio) % HOP_LENGTH)]
        self.audio_dataset = None

    def setup(self, stage=None):
        self.audio_dataset = AudioDataset()

    def train_dataloader(self):
        return DataLoader(self.audio_dataset)

    def val_dataloader(self):
        return DataLoader(self.audio_dataset)

    def test_dataloader(self):
        return DataLoader(self.audio_dataset)

    def predict_dataloader(self):
        return DataLoader(self.audio_dataset)


cli = LightningCLI(DDSP, AudioDataModule)
