"""
Preprocessing Step 3: Make Manifest csv files from chunks for training/validation
    This is the final step
"""
import pandas as pd
import os
import numpy as np
import glob
import json
import argparse
import tqdm

parser = argparse.ArgumentParser()
parser.description = "Script to make experiment manifests"
parser.add_argument("--dev_csv", type=str, default=None,
                    help="path to dev.csv file found in FSD50K.ground_truth")
parser.add_argument("--eval_csv", type=str, default=None,
                    help="path to eval.csv file found in FSD50K.ground_truth")
parser.add_argument("--dev_chunks_dir", type=str,
                    help="path to directory containing dev chunks")
parser.add_argument("--eval_chunks_dir", type=str, default=None,
                    help="path to directory containing eval chunks")
parser.add_argument("--output_dir", type=str,
                    help="path to directory for storing manifest csv files used for training")


if __name__ == '__main__':
    args = parser.parse_args()

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    process_eval = False if args.eval_csv is None else True
    process_dev = False if args.dev_csv is None else True

    if process_dev:
        all_df = pd.read_csv(args.dev_csv)
        tr = all_df[all_df['split'] == 'train']
        val = all_df[all_df['split'] == 'val']
        tr_files = tr['fname'].values
        tr_labels = tr['labels'].values

        val_files = val['fname'].values
        val_labels = val['labels'].values

        dev_chunks_dir = args.dev_chunks_dir

        tr_lbl_map = {}
        for i in tqdm.tqdm(range(len(tr_files))):
            ext = tr_files[i]
            lbl = tr_labels[i]
            # ext = f.split("/")[-1].split(".")[0]
            tr_lbl_map[str(ext)] = lbl

        val_lbl_map = {}
        for i in tqdm.tqdm(range(len(val_files))):
            ext = val_files[i]
            lbl = val_labels[i]
            # ext = f.split("/")[-1].split(".")[0]
            val_lbl_map[str(ext)] = lbl

        tr_chunk_labels = []
        tr_chunk_exts = []
        tr_chunk_files = []

        val_chunk_labels = []
        val_chunk_exts = []
        val_chunk_files = []

        dev_chunk_files = glob.glob(os.path.join(dev_chunks_dir, "*.wav"))
        tr_keys = list(tr_lbl_map.keys())
        val_keys = list(val_lbl_map.keys())

        for f in tqdm.tqdm(dev_chunk_files):
            ext = f.split("/")[-1].split(".")[0].split("_")[0]
            if ext in val_keys:
                val_chunk_labels.append(val_lbl_map[ext])
                val_chunk_exts.append(ext)
                val_chunk_files.append(f)
            else:
                tr_chunk_labels.append(tr_lbl_map[ext])
                tr_chunk_exts.append(ext)
                tr_chunk_files.append(f)
        tr_chunk_labels = np.asarray(tr_chunk_labels)
        tr_chunk_exts = np.asarray(tr_chunk_exts)
        tr_chunk_files = np.asarray(tr_chunk_files)

        val_chunk_labels = np.asarray(val_chunk_labels)
        val_chunk_exts = np.asarray(val_chunk_exts)
        val_chunk_files = np.asarray(val_chunk_files)

        tr_chunk = pd.DataFrame()
        tr_chunk['files'] = tr_chunk_files
        tr_chunk['labels'] = tr_chunk_labels
        tr_chunk['ext'] = tr_chunk_exts
        tr_chunk = tr_chunk.iloc[np.random.permutation(len(tr_chunk))]

        val_chunk = pd.DataFrame()
        val_chunk['files'] = val_chunk_files
        val_chunk['labels'] = val_chunk_labels
        val_chunk['ext'] = val_chunk_exts
        val_chunk = val_chunk.sort_values(['ext'])

        tr_chunk.to_csv(os.path.join(args.output_dir, "tr.csv"), index=False)
        val_chunk.to_csv(os.path.join(args.output_dir, "val.csv"), index=False)

        vocab = pd.read_csv(args.dev_csv.replace("dev.csv", "vocabulary.csv"), names=['ix', 'label_name', 'label_id'])
        lbl_map = {}
        for ix in range(len(vocab)):
            rec = vocab.iloc[ix]
            lbl_map[rec['label_name']] = int(rec['ix'])
        with open(os.path.join(args.output_dir, "lbl_map.json"), "w") as fd:
            json.dump(lbl_map, fd)

    if process_eval:
        eval_chunks_dir = args.eval_chunks_dir
        eval_df = pd.read_csv(args.eval_csv)
        ext_lbl_map_eval = {}
        all_eval_files = eval_df['fname'].values
        all_eval_labels = eval_df['labels'].values
        for i in tqdm.tqdm(range(len(all_eval_files))):
            ext = all_eval_files[i]
            lbl = all_eval_labels[i]
            # ext = f.split("/")[-1].split(".")[0]
            # ext =
            ext_lbl_map_eval[str(ext)] = lbl
        eval_chunk_files = glob.glob(os.path.join(eval_chunks_dir, "*.wav"))
        eval_chunk_labels = []
        eval_chunk_exts = []
        eval_files = []
        for f in tqdm.tqdm(eval_chunk_files):
            ext = f.split("/")[-1].split(".")[0].split("_")[0]
            eval_chunk_labels.append(ext_lbl_map_eval[ext])
            eval_chunk_exts.append(ext)
            eval_files.append(f)
        eval_chunk_labels = np.asarray(eval_chunk_labels)
        eval_chunk_exts = np.asarray(eval_chunk_exts)
        eval_files = np.asarray(eval_files)
        eval_chunk = pd.DataFrame()
        eval_chunk['files'] = eval_files
        eval_chunk['labels'] = eval_chunk_labels
        eval_chunk['ext'] = eval_chunk_exts
        # eval_chunk = eval_chunk.iloc[np.random.permutation(len(eval_chunk))]
        eval_chunk = eval_chunk.sort_values(['ext'])
        # eval_chunk.to_csv("/media/sarthak/nvme/datasets/fsd50k/meta_chunks/eval.csv", index=False)
        eval_chunk.to_csv(os.path.join(args.output_dir, "eval.csv"), index=False)
