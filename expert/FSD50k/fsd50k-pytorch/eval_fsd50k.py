import os
import tqdm
import numpy as np
import torch
import argparse
from src.models.fsd50k_lightning import FSD50k_Lightning
from src.data.fsd_eval_dataset import FSD50kEvalDataset, _collate_fn_eval
from src.utilities.metrics_helper import calculate_stats, d_prime

parser = argparse.ArgumentParser()
parser.description = "Evaluation script for FSD50k Baselines"
parser.add_argument("--ckpt_path", type=str,
                    help="path to model .ckpt")
parser.add_argument("--eval_csv", type=str,
                    help="path to eval.csv manifest file")
parser.add_argument("--lbl_map", type=str,
                    help="path to label map .json file")


if __name__ == '__main__':
    args = parser.parse_args()
    model = FSD50k_Lightning.load_from_checkpoint(args.ckpt_path)
    model = model.cuda().eval()

    test_set = FSD50kEvalDataset(args.eval_csv, args.lbl_map,
                                 model.hparams.cfg['audio_config'],
                                 transform=model.hparams.val_tfs)
    cnt = 0
    test_predictions = []
    test_gts = []
    for ix in tqdm.tqdm(range(test_set.len)):
        with torch.no_grad():
            batch = test_set[ix]
            x, y = batch
            x = x.cuda()
            y_pred = model(x)
            y_pred = y_pred.mean(0).unsqueeze(0)
            sigmoid_preds = torch.sigmoid(y_pred)
        test_predictions.append(sigmoid_preds.detach().cpu().numpy()[0])
        test_gts.append(y.detach().cpu().numpy()[0])        # drop batch axis
    test_predictions = np.asarray(test_predictions).astype('float32')
    test_gts = np.asarray(test_gts).astype('int32')
    stats = calculate_stats(test_predictions, test_gts)
    mAP = np.mean([stat['AP'] for stat in stats])
    mAUC = np.mean([stat['auc'] for stat in stats])
    print("mAP: {:.6f}".format(mAP))
    print("mAUC: {:.6f}".format(mAUC))
    print("dprime: {:.6f}".format(d_prime(mAUC)))
