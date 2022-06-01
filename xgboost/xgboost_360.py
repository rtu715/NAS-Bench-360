import fire
import numpy as np
from perceiver.data import *
from perceiver.data.nb360 import darcy_utils
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    f1_score,
    average_precision_score,
)
from functools import partial

import utils
import xgboost as xgb

import torch
import torchmetrics as tm
import pytorch_lightning as pl


def main(task="cifar100", seed=7734, load_np=False, save_np=False, no_test=False):  # 😈
    # Set various random seeds
    # pl.utilities.seed.seed_everything(seed=seed, workers=True)

    # Common model params
    model_params = {
        "random_state": seed,
        "max_depth": 3,
        "eta": 1,
        "n_jobs": 10,
        "gpu_id": 0,
        "early_stopping_rounds": 5,
        "tree_method": "gpu_hist",
        "subsample": 0.9,
        "sampling_method": "gradient_based",
    }

    if task == "cifar100":
        dm = CIFAR100DataModule(batch_size=10_000, root="datasets")
        model_params = {
            "objective": "multi:softmax",
            "eval_metric": "merror",
            "num_class": 100,
            **model_params,
        }
        fit_params = {
            "verbose": True,
        }
        model = xgb.XGBClassifier(**model_params)
        eval_metric = accuracy_score
    elif task == "spherical":
        dm = SphericalDataModule(batch_size=10_000, root="datasets")
        model_params = {
            "objective": "multi:softmax",
            "eval_metric": "merror",
            "num_class": 100,
            **model_params,
        }
        fit_params = {
            "verbose": True,
        }
        model = xgb.XGBClassifier(**model_params)
        eval_metric = accuracy_score
    elif task == "ninapro":
        dm = NinaProDataModule(batch_size=10_000, root="datasets")
        model_params = {
            "objective": "multi:softmax",
            "eval_metric": "merror",
            "num_class": 18,
            **model_params,
        }
        print(model_params)
        fit_params = {
            "verbose": True,
        }
        model = xgb.XGBClassifier(**model_params)
        eval_metric = accuracy_score
    elif task == "satellite":
        dm = SatelliteDataModule(batch_size=10_000, root="datasets")
        model_params = {
            "objective": "multi:softmax",
            "eval_metric": "merror",
            "num_class": 24,
            **model_params,
        }
        fit_params = {
            "verbose": True,
        }
        model = xgb.XGBClassifier(**model_params)
        eval_metric = accuracy_score
    elif task == "deepsea":
        dm = DeepSEADataModule(batch_size=10_000, root="datasets")
        model_params = {
            **model_params,
        }
        fit_params = {
            "verbose": True,
        }
        model = xgb.XGBClassifier(**model_params)
        eval_metric = roc_auc_score
    elif task == "ecg":
        dm = ECGDataModule(batch_size=10_000, root="datasets")
        model_params = {
            "objective": "binary:logistic",
            **model_params,
        }
        fit_params = {
            "verbose": True,
        }
        model = xgb.XGBClassifier(**model_params)
        eval_metric = partial(f1_score, average="macro")
    elif task == "fsd50k":
        dm = FSD50KDataModule(batch_size=100, root="datasets")
        model_params = {
            **model_params,
        }
        fit_params = {
            "verbose": True,
        }
        model = xgb.XGBClassifier(**model_params)
    elif task == "darcyflow":
        dm = DarcyFlowDataModule(batch_size=100, root="datasets")
        model_params = {
            **model_params,
        }
        fit_params = {
            "verbose": True,
        }
        model = xgb.XGBRegressor(**model_params)
        eval_metric = darcy_utils.LpLoss(size_average=False)
    elif task == "cosmic":
        dm = CosmicDataModule(batch_size=100, root="datasets")
        model_params = {
            **model_params,
        }
        fit_params = {
            "verbose": True,
        }
        model = xgb.XGBClassifier(**model_params)
        eval_metric = None
    elif task == "psicov":
        dm = PSICOVDataModule(batch_size=100, root="datasets")
        model_params = {
            **model_params,
        }
        model_params['tree_method'] = 'hist'
        model_params['sampling_method'] = 'uniform'
        fit_params = {
            "verbose": True,
        }
        model = xgb.XGBRegressor(**model_params)
        eval_metric = None
    else:
        raise NotImplementedError

    if load_np:
        with open(f"{task}_train.npy", "rb") as f:
            x_train = np.load(f)
            y_train = np.load(f)
            r = np.random.permutation(x_train.shape[0])
            x_train = x_train[r]
            y_train = y_train[r]
        with open(f"{task}_valid.npy", "rb") as f:
            x_valid = np.load(f)
            y_valid = np.load(f)
        with open(f"{task}_test.npy", "rb") as f:
            x_test = np.load(f)
            y_test = np.load(f)
    else:
        dm.setup(stage=None)
        x_train, y_train = utils.dm_to_numpy(dm.train_dataloader())
        x_valid, y_valid = utils.dm_to_numpy(dm.val_dataloader())
        x_test, y_test = utils.dm_to_numpy(dm.test_dataloader())
        if save_np:
            with open(f"{task}_train.npy", "wb") as f:
                np.save(f, x_train)
                np.save(f, y_train)
            with open(f"{task}_valid.npy", "wb") as f:
                np.save(f, x_valid)
                np.save(f, y_valid)
            with open(f"{task}_test.npy", "wb") as f:
                np.save(f, x_test)
                np.save(f, y_test)

    model.fit(
        x_train,
        y_train,
        eval_set=[(x_train, y_train), (x_valid, y_valid)],
        **fit_params,
    )
    model.save_model(f"xgboost_model_{task}_{seed}.json")

    if not no_test:
        if task != "fsd50k" and task != "darcyflow":
            y_test_preds = model.predict(x_test)
            test_score = eval_metric(y_test, y_test_preds)
            print(f"test score: {test_score}")

        elif task == "fsd50k":
            y_test_preds = model.predict(x_test)
            y_test = torch.tensor(np.array(y_test))
            y_test_preds = torch.tensor(np.array(y_test_preds))
            eval_metric = tm.AveragePrecision(pos_label=1, average="macro")
            test_score = eval_metric(y_test, y_test_preds)
            print(f"test score: {test_score}")

        elif task == "darcyflow":
            y_test_preds = model.predict(x_test)
            y_test = torch.tensor(y_test)
            y_test_preds = torch.tensor(y_test_preds)
            test_score = eval_metric(y_test, y_test_preds) / 100.0
            print(f"test score: {test_score}")


if __name__ == "__main__":
    fire.Fire(main)
