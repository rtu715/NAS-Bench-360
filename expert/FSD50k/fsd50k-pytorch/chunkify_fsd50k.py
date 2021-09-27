"""
Preprocessing Step 2:
    For making overlapping chunks of audio files
    as specified in FSD50k paper [https://arxiv.org/abs/2010.00475], pg 16, B. Baseline Systems

    optionally: can be used to precompute spectrograms instead
"""
import numpy as np
import os
import glob
from multiprocessing import Pool
import soundfile as sf
import argparse

parser = argparse.ArgumentParser()
parser.description = "Multiprocessing script for making chunks as per FSD50K guidelines"
parser.add_argument("--src_dir", "-s", type=str,
                    help="source directory containing .wav files")
parser.add_argument("--tgt_dir", "-t", type=str,
                    help="target directory where chunks with be stored")

args = parser.parse_args()

files = glob.glob("{}/*.wav".format(args.src_dir))
tgt_dir = args.tgt_dir

lf = len(files)

if not os.path.exists(tgt_dir):
    os.makedirs(tgt_dir)


def replicate_if_needed(x, min_clip_duration):
    if len(x) < min_clip_duration:
        tile_size = (min_clip_duration // x.shape[0]) + 1
        x = np.tile(x, tile_size)[:min_clip_duration]
    return x


def process_idx(idx):
    f = files[idx]
    fname = f.split("/")[-1].split(".")[0]
    x, sr = sf.read(f)
    min_clip_duration = int(sr * 1)
    parts = []
    if len(x) < min_clip_duration:
        x = replicate_if_needed(x, min_clip_duration)
        parts.append(x)
    else:
        overlap = int(sr * 0.5)
        for ix in range(0, len(x), overlap):
            clip_ix = x[ix:ix+min_clip_duration]
            clip_ix = replicate_if_needed(clip_ix, min_clip_duration)
            parts.append(clip_ix)

    for jx in range(len(parts)):
        pth = os.path.join(tgt_dir, "{}_{:04d}.wav".format(fname, jx))
        sf.write(pth, parts[jx], sr, "PCM_16")

    if idx % 500 == 0:
        print("Done: {:07d}/{:07d}".format(idx, lf))


if __name__ == '__main__':
    pool = Pool(24)
    o = pool.map_async(process_idx, range(lf))
    res = o.get()
    pool.close()
    pool.join()
