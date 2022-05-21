import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F  # noqa
from einops.layers.torch import Reduce
from einops import rearrange
import torch.fft as fft
from constants import *


class HarmonicOscillator(nn.Module):
    def __init__(self, n_harmonics: int = 64, n_channels: int = 1):
        super().__init__()

        self.n_harmonics = n_harmonics
        self.n_channels = n_channels

        harmonics = torch.arange(1, self.n_harmonics + 1, step=1)
        self.register_buffer("harmonics", harmonics, persistent=False)

        self.sum_sinusoids = Reduce("b c o t -> b c t", "sum")

    def forward(self, f0: torch.Tensor, master_amplitude: torch.Tensor, overtone_amplitudes: torch.Tensor):
        # f0.shape = [batch, n_channels, time]
        # master_amplitude.shape = [batch, n_channels, time]
        # overtone_amplitudes = [batch, n_channels, n_harmonics, time]

        # Calculate overtone frequencies
        overtone_fs = torch.einsum("bct,o->bcot", f0, self.harmonics)
        # Convert overtone frequencies from Hz to radians / sample
        overtone_fs *= 2 * np.pi
        overtone_fs /= SAMPLE_RATE

        # set amplitudes of overtones above Nyquist to 0.0
        overtone_amplitudes[overtone_fs > np.pi] = 0.0
        # normalize harmonic_distribution so it always sums to one
        overtone_amplitudes /= torch.sum(overtone_amplitudes, dim=2, keepdim=True)
        # scale individual overtone amplitudes by the master amplitude
        overtone_amplitudes = torch.einsum("bcot,bct->bcot", overtone_amplitudes, master_amplitude)

        # stretch controls by hop_size
        # refactor stretch into a function or a method
        # overtone_fs = self.pre_stretch(overtone_fs)
        overtone_fs = F.interpolate(overtone_fs, size=(overtone_fs.shape[-2], (f0.shape[-1] - 1) * HOP_LENGTH),
                                    mode='bilinear', align_corners=True)
        # overtone_fs = self.post_stretch(overtone_fs)
        # overtone_amplitudes = self.pre_stretch(overtone_amplitudes)
        overtone_amplitudes = F.interpolate(overtone_amplitudes,
                                            size=(overtone_amplitudes.shape[-2], (f0.shape[-1] - 1) * HOP_LENGTH),
                                            mode='bilinear', align_corners=True)
        # overtone_amplitudes = self.post_stretch(overtone_amplitudes)

        # calculate phases and sinusoids
        # TODO: randomizing phases. Is it necessary?
        overtone_fs[:, :, :, 0] = 3.14159265 * (torch.rand(*overtone_fs.shape[:-1], device=overtone_fs.device) * 2 - 1)
        phases = torch.cumsum(overtone_fs, dim=-1)
        sinusoids = torch.sin(phases)

        # scale sinusoids by their corresponding amplitudes and sum them to get the final signal
        sinusoids = torch.einsum("bcot,bcot->bcot", sinusoids, overtone_amplitudes)
        signal = self.sum_sinusoids(sinusoids)

        return signal


class FilteredNoise(nn.Module):
    def __init__(self, n_bands: int = 128, n_channels: int = 1, ):
        super().__init__()

        self.n_bands = n_bands
        self.n_channels = n_channels

    def forward(self, filter_bands):
        # filter_bands.shape = [batch, n_channels, n_bands, time]

        # Generate white noise
        batch_size, _, _, n_steps = filter_bands.shape
        # noise.shape = [batch, n_channels, time]
        noise = torch.rand(batch_size, self.n_channels, (n_steps - 1) * HOP_LENGTH, device=filter_bands.device) * 2 - 1

        # Get frames
        padded_noise = F.pad(noise, (N_FFT//2, N_FFT//2))
        # noise_frames.shape = [batch, n_channels, n_sample (window_length), n_frames (time)]
        noise_frames = padded_noise.unfold(-1, N_FFT, HOP_LENGTH)

        # Stretch filter to window_length // 2
        filter_ = rearrange(filter_bands, "b c f t -> (b t) c f")
        filter_ = F.interpolate(filter_, size=N_FFT // 2, mode='nearest')
        filter_ = rearrange(filter_, "(b t) c f -> b c t f", b=batch_size, t=n_steps)

        # Prepend 0 DC offset
        dc = torch.zeros(*filter_.shape[:-1], 1).to(filter_.device)
        filter_ = torch.concat([dc, filter_], dim=-1)

        # apply filter to noise
        fft_noise_frames = fft.rfft(noise_frames)
        filtered_fft_noise_frames = filter_ * fft_noise_frames
        filtered_noise_frames = fft.irfft(filtered_fft_noise_frames)
        filtered_noise_frames *= torch.hann_window(N_FFT, periodic=False, device=filter_.device)

        # overlap add
        # I forgot what I have done here, but it seems to work
        b, c = filtered_noise_frames.shape[0], filtered_noise_frames.shape[1]
        stacked_noise = rearrange(filtered_noise_frames, "b c t f -> (b c) f t")
        filtered_noise = F.fold(
            stacked_noise, (1, padded_noise.shape[-1]), (1, N_FFT), stride=(1, HOP_LENGTH)
        )
        filtered_noise = rearrange(filtered_noise, "(b c) 1 1 t -> b c t", b=b, c=c)

        # remove padding and return
        return filtered_noise[:, :, N_FFT // 2:-N_FFT // 2]


def pad_to(tensor: torch.Tensor, target_length: int, mode: str = 'constant', value: float = 0):
    """
    Pad the given tensor to the given length, with 0s on the right.
    """
    return F.pad(tensor, (0, target_length - tensor.shape[-1]), mode=mode, value=value)


def unfold(x, kernel_size: int, stride: int):
    """1D only unfolding similar to the one from PyTorch.
    However, PyTorch unfold is extremely slow.
    Given an input tensor of size `[*, T]` this will return
    a tensor `[*, F, K]` with `K` the kernel size, and `F` the number
    of frames. The i-th frame is a view onto `i * stride: i * stride + kernel_size`.
    This will automatically pad the input to cover at least once all entries in `input`.
    Args:
        x (Tensor): tensor for which to return the frames.
        kernel_size (int): size of each frame.
        stride (int): stride between each frame.
    Shape:
        - Inputs: `input` is `[*, T]`
        - Output: `[*, F, kernel_size]` with `F = 1 + ceil((T - kernel_size) / stride)`
    ..Warning:: unlike PyTorch unfold, this will pad the input
        so that any position in `input` is covered by at least one frame.
    """
    shape = list(x.shape)
    length = shape.pop(-1)
    n_frames = math.ceil((max(length, kernel_size) - kernel_size) / stride) + 1
    tgt_length = (n_frames - 1) * stride + kernel_size
    padded = F.pad(x, (0, tgt_length - length)).contiguous()
    strides = []
    for dim in range(padded.dim()):
        strides.append(padded.stride(dim))
    assert strides.pop(-1) == 1, 'data should be contiguous'
    strides = strides + [stride, 1]
    return padded.as_strided(shape + [n_frames, kernel_size], strides)


def fft_conv1d(
        x: torch.Tensor, weight: torch.Tensor,
        bias=None, stride: int = 1, padding: int = 0,
        block_ratio: float = 5):
    """
    Same as `torch.nn.functional.conv1d` but using FFT for the convolution.
    Please check PyTorch documentation for more information.
    Args:
        x (Tensor): input signal of shape `[B, C, T]`.
        weight (Tensor): weight of the convolution `[D, C, K]` with `D` the number
            of output channels.
        bias (Tensor or None): if not None, bias term for the convolution.
        stride (int): stride of convolution.
        padding (int): padding to apply to the input.
        block_ratio (float): can be tuned for speed. The input is split in chunks
            with a size of `int(block_ratio * kernel_size)`.
    Shape:
        - Inputs: `input` is `[B, C, T]`, `weight` is `[D, C, K]` and bias is `[D]`.
        - Output: `(*, T)`
    ..note::
        This function is faster than `torch.nn.functional.conv1d` only in specific cases.
        Typically, the kernel size should be of the order of 256 to see any real gain,
        for a stride of 1.
    ..Warning::
        Dilation and groups are not supported at the moment. This function might use
        more memory than the default Conv1d implementation.
    """
    x = F.pad(x, (padding, padding))
    batch, channels, length = x.shape
    out_channels, _, kernel_size = weight.shape

    if length < kernel_size:
        raise RuntimeError(f"Input should be at least as large as the kernel size {kernel_size}, "
                           f"but it is only {length} samples long.")
    if block_ratio < 1:
        raise RuntimeError("Block ratio must be greater than 1.")

    # We are going to process the input blocks by blocks, as for some reason it is faster
    # and less memory intensive (I think the culprit is `torch.einsum`).
    block_size: int = min(int(kernel_size * block_ratio), length)
    fold_stride = block_size - kernel_size + 1
    weight = pad_to(weight, block_size)
    weight_z = fft.rfft(weight)

    # We pad the input and get the different frames, on which
    frames = unfold(x, block_size, fold_stride)

    frames_z = fft.rfft(frames)
    out_z = frames_z * weight_z.conj()
    out = fft.irfft(out_z, block_size)
    # The last bit is invalid, because FFT will do a circular convolution.
    out = out[..., :-kernel_size + 1]
    out = out.reshape(batch, out_channels, -1)
    out = out[..., ::stride]
    target_length = (length - kernel_size) // stride + 1

    # TODO: this line throws away the tail. Will be necessary for real-time synth.
    out = out[..., :target_length]
    if bias is not None:
        out += bias[:, None]
    return out


def init_ir(duration=3, in_ch=1, out_ch=1):
    length = duration * SAMPLE_RATE - 1
    ir = torch.randn(out_ch, in_ch, length)
    envelop = torch.exp(-4.0 * torch.linspace(0., duration, steps=length))
    ir *= envelop
    ir = ir / torch.norm(ir, p=2, dim=-1, keepdim=True)
    ir = ir.flip(-1)

    # we can train a mono synth and controller, add stereo width using reverb
    # [output dimension, input_dimension, time]
    return ir


class ConvolutionalReverb(nn.Module):
    def __init__(self, duration=3, in_ch=1, out_ch=1, block_ratio=5):
        super().__init__()
        if block_ratio < 1:
            raise RuntimeError("Block ratio must be greater than 1.")
        self.block_ratio = block_ratio
        # first, (if reversed, last) bit of ir should always be 1
        self.ir = nn.Parameter(init_ir(duration, in_ch, out_ch))

    def forward(self, x: torch.Tensor):
        ir = torch.concat([self.ir, torch.ones(*self.ir.shape[:-1], 1, device=x.device)], dim=-1)
        x = torch.nn.functional.pad(x, (ir.shape[-1] - 1, 0))
        out = fft_conv1d(x, self.ir, block_ratio=self.block_ratio)

        return out
