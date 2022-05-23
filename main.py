import torch
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

        if batch_nb == 0:
            write_wav('./sick.wav', y[0].detach().cpu().numpy().T, SAMPLE_RATE)

        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)


class AudioDataset(Dataset):
    def __init__(self):
        super().__init__()
        self.audio = torch.load(CELLO_AUDIO_DIR.parent / 'audio.pth')
        self.loudness = torch.load(CELLO_AUDIO_DIR.parent / 'loudness.pth')
        self.f0 = torch.load(CELLO_AUDIO_DIR.parent / 'f0.pth')

    def __len__(self):
        return self.audio.shape[0]

    def __getitem__(self, idx):
        return self.f0[idx], self.loudness[idx], self.audio[idx]


class AudioDataModule(LightningDataModule):
    def __init__(self):
        super().__init__()
        self.audio_dataset = None

    def setup(self, stage=None):
        self.audio_dataset = AudioDataset()

    def train_dataloader(self):
        return DataLoader(self.audio_dataset, batch_size=8, shuffle=True, num_workers=6)

    def val_dataloader(self):
        return DataLoader(self.audio_dataset)

    def test_dataloader(self):
        return DataLoader(self.audio_dataset)

    def predict_dataloader(self):
        return DataLoader(self.audio_dataset)


cli = LightningCLI(DDSP, AudioDataModule)
