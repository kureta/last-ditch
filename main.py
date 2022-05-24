import torch
from pytorch_lightning.utilities.cli import LightningCLI

from torch.utils.data import Dataset, DataLoader
from pytorch_lightning import LightningDataModule, LightningModule

from dsp import HarmonicOscillator, FilteredNoise, ConvolutionalReverb
from loss import distance
from model import Controller
# from soundfile import write as wav_write
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

        self.log('train/loss', loss)

        return loss

    def validation_step(self, batch, batch_nb):
        f0, amp, x = batch
        with torch.no_grad():
            y = self(f0, amp)
            loss = distance(x, y)

        self.log('val/loss', loss)
        if batch_nb < 4:
            self.logger.experiment.add_audio(
                f'{batch_nb}',
                y[0, 0],
                self.global_step,
                sample_rate=SAMPLE_RATE
            )
            # wav_write(f'{batch_nb}.wav', y[0].cpu().numpy().T, SAMPLE_RATE)

        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)


class AudioDataset(Dataset):
    def __init__(self, path):
        super().__init__()
        self.features = torch.load(path)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx]['f0'], self.features[idx]['loudness'], self.features[idx]['audio']


class AudioDataModule(LightningDataModule):
    def __init__(self, batch_size=8, path=CELLO_AUDIO_DIR.parent / 'features.pth'):
        super().__init__()
        self.train_dataset = None
        self.val_dataset = None
        self.batch_size = batch_size
        self.path = path

    def setup(self, stage=None):
        self.train_dataset = AudioDataset(self.path)
        self.val_dataset = AudioDataset(self.path)
        self.val_dataset.features = self.val_dataset.features[:64]

    def train_dataloader(self):
        return DataLoader(self.train_dataset,
                          batch_size=self.batch_size, shuffle=True, num_workers=4,
                          pin_memory=True, persistent_workers=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset,
                          batch_size=self.batch_size, shuffle=False, num_workers=4,
                          pin_memory=False, persistent_workers=False)


def cli_main():
    cli = LightningCLI(DDSP, AudioDataModule)


if __name__ == '__main__':
    cli_main()
