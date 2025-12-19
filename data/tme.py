import os
import pandas as pd

import yaml
import torch
import torchaudio
from pathlib import Path
from torchcomp import amp2db
import pyloudnorm as pyln


def find_time_offset(x: torch.Tensor, y: torch.Tensor):
    x = x.double()
    y = y.double()
    N = x.size(-1)
    M = y.size(-1)

    X = torch.fft.rfft(x, n=N + M - 1)
    Y = torch.fft.rfft(y, n=N + M - 1)
    corr = torch.fft.irfft(X.conj() * Y)
    shifts = torch.argmax(corr, dim=-1)
    return torch.where(shifts >= N, shifts - N - M + 1, shifts)


def tme_vocal(
    song_id, loudness: float = -18.0, return_data=True, side_energy_threshold=-10
):  
    print(os.listdir(song_id), song_id)
    print(os.listdir(os.path.join(song_id, 'render')))
    print(os.listdir(os.path.join(song_id, 'stereo')))
    wet_subdir = "render" # ['vocals.wav', 'acc.wav']
    dry_subdir = "stereo" # ['vocals.wav', 'acc.wav']
    song_id = Path(song_id)

    results = []
    wet_file = song_id / wet_subdir / 'vocals.wav'
    dry_file = song_id / dry_subdir / 'vocals.wav'

    if not return_data:
        results.append((dry_file, wet_file))
        return results

    dry, sr = torchaudio.load(str(dry_file))
    if dry.size(0) > 1:
        left = dry[0]
        right = dry[1]
        left = left / left.max()
        right = right / right.max()
        side = (left - right) * 0.707
        side_energy = amp2db(side.abs().max()).item()
    else:
        side_energy = -torch.inf

    # print(f"Maximum energy of side channel: {side_energy:.4f} dB")
    # if side_energy > side_energy_threshold:
    #     print(f"Skip {dry_file}")
    #     return results

    wet, _ = torchaudio.load(str(wet_file))
    assert sr == _

    dry = dry[:, : wet.shape[1]]
    wet = wet[:, : dry.shape[1]]

    shifts = find_time_offset(dry.mean(0), wet.mean(0)).item()
    dry = torch.roll(dry, shifts=int(shifts), dims=1)
    print(shifts, dry.shape)

    dry = dry.mean(0, keepdim=True)

    meter = pyln.Meter(sr)
    normaliser = lambda x: pyln.normalize.loudness(
        x, meter.integrated_loudness(x), loudness
    )
    dry = torch.from_numpy(normaliser(dry.numpy().T).T).float()
    wet = torch.from_numpy(normaliser(wet.numpy().T).T).float()

    results.append((dry_file, wet_file, sr, dry, wet, shifts))
    return results


if __name__ == "__main__":
    dataset_dir = '/cfs3/rogerhhuang/spatial_audio/'
    split = 'data_pair-vocals_acc'
    song_id = '105016836'
    
    # song_list = os.listdir(os.path.join(dataset_dir, split))
    # song_list = [os.path.join(dataset_dir, split, s) for s in song_list]
    # fname = f'{split}_file_list.csv'
    # df = pd.DataFrame(song_list)
    # df.to_csv(fname, index=False, header=False)

    song_list = pd.read_csv(f'{split}_file_list.csv', header=None)[0].tolist()
    print(f'Total songs: {len(song_list)}')

    for song_id in song_list:
        results = tme_vocal(
            song_id, loudness=-18.0)
        break
