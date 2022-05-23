import math

import librosa
import torch
import torchaudio.functional as AF  # noqa
import torch.nn.functional as F  # noqa
import torchcrepe
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from tqdm.contrib.concurrent import process_map

from constants import *


def load_and_resample(f: Path):
    audio, sr = librosa.load(f, sr=SAMPLE_RATE, mono=False)
    audio = torch.from_numpy(audio.astype('float32'))

    return audio


def prepare_audio(samples_dir: Path):
    files = list(samples_dir.glob('*.wav'))
    files.sort()

    sample_duration = 48000 * 4  # 4 seconds

    audios = process_map(load_and_resample, files, max_workers=8)
    audios = torch.cat(audios, dim=1)
    audios = audios.unfold(-1, sample_duration, sample_duration // 2).transpose(0, 1)

    torch.save(audios, samples_dir.parent / 'audio.pth')


freqs = librosa.core.fft_frequencies(sr=SAMPLE_RATE, n_fft=N_FFT)


def get_loudness(x: torch.Tensor):
    # x = [batch, time] mono signal without channel dimension
    S = torch.stft(
        x,
        N_FFT,
        HOP_LENGTH,
        N_FFT,
        torch.hann_window(N_FFT).to(x),
        center=True,
        normalized=False,
        return_complex=True,
    ).abs()

    # S = [batch, freq, time]
    Sdb = AF.amplitude_to_DB(S, 20., 1e-5, 0., 80.)
    weights = torch.from_numpy(librosa.A_weighting(freqs)[None, :, None].astype('float32')).to(Sdb)
    Sdb += weights

    loudness = torch.mean(Sdb, dim=1)  # take mean over freqs to get loudness

    # loudness = [batch, time]
    return loudness


class AudioDataset(Dataset):
    def __init__(self, data_path: Path):
        super().__init__()
        self.audio = torch.load(data_path)

    def __len__(self):
        return self.audio.shape[0]

    def __getitem__(self, idx):
        return self.audio[idx]


def prepare_loudness(samples_dir: Path, batch_size=32):
    audio = AudioDataset(samples_dir.parent / 'audio.pth')
    loader = DataLoader(audio, batch_size=batch_size, shuffle=False)
    loudness = []
    for example in tqdm(loader, total=math.ceil(len(audio) / batch_size)):
        example = torch.mean(example, dim=1)
        loudness.append(get_loudness(example.cuda()).cpu())

    loudness = torch.cat(loudness, dim=0)[:, None, :]

    torch.save(loudness, samples_dir.parent / 'loudness.pth')


def prepare_f0(samples_dir: Path):
    audio = AudioDataset(samples_dir.parent / 'audio.pth')
    loader = DataLoader(audio, batch_size=1, shuffle=False)

    f0s = []
    for example in tqdm(loader, total=len(audio)):
        example = torch.mean(example, dim=1)
        example = AF.resample(example, SAMPLE_RATE, 16000)
        f0, _, probs = torchcrepe.predict(example,
                                          sample_rate=16000,
                                          hop_length=HOP_LENGTH // 3,
                                          fmin=50.,
                                          decoder=torchcrepe.decode.weighted_argmax,
                                          device='cuda', return_periodicity=True)
        if torch.isnan(f0).any():
            torch.save(example, 'bad.pth')
            torch.save(f0, 'fbad.pth')
            print('something is fishy')
            exit(1)
        f0s.append(f0.cpu())

    f0s = torch.cat(f0s, dim=0)[:, None, :]
    torch.save(f0s, samples_dir.parent / 'f0.pth')


def main():
    # prepare_audio(CELLO_AUDIO_DIR)
    prepare_loudness(CELLO_AUDIO_DIR)
    prepare_f0(CELLO_AUDIO_DIR)


if __name__ == '__main__':
    main()
