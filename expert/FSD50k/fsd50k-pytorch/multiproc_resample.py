"""
Preprocessing Step 1: Convert audio files into 22050 Hz audio
"""
import subprocess as sp
import os
import glob
from multiprocessing import Pool
import argparse

parser = argparse.ArgumentParser()
parser.description = "Multi-processing resampling script"
parser.add_argument("--src_path", type=str,
                    help="path to source directory containing .wav files")
parser.add_argument("--dst_path", type=str,
                    help="path to destination directory where resampled files will be stored")
parser.add_argument("--sample_rate", type=int, default=22050,
                    help="target sample rate")

args = parser.parse_args()

files = glob.glob(os.path.join(args.src_path, "*.wav"))
tgt_dir = args.dst_path
if not os.path.exists(tgt_dir):
    os.makedirs(tgt_dir)

lf = len(files)
SAMPLE_RATE = args.sample_rate


def process_idx(idx):
    f = files[idx]
    fname = f.split("/")[-1]
    tgt_path = os.path.join(tgt_dir, fname)
    command = "ffmpeg -loglevel 0 -nostats -i '{}' -ac 1 -ar {} '{}'".format(f, SAMPLE_RATE, tgt_path)
    sp.call(command, shell=True)
    if idx % 500 == 0:
        print("Done: {:05d}/{}".format(idx, lf))


if __name__ == '__main__':
    pool = Pool(6)
    o = pool.map_async(process_idx, range(lf))
    res = o.get()
    pool.close()
    pool.join()
