import fire
import numpy as np
from perceiver.data import *
from sklearn.metrics import accuracy_score

import utils
import xgboost as xgb


def main(task="cifar100", seed=7734):  # 😈
    # TODO set various random seeds

    # Common model params
    model_params = {
        "seed": seed,
        "max_depth": 3,
        "eta": 1,
        "n_jobs": 10,
        "gpu_id": 0,
        "early_stopping_rounds": 5,
        "tree_method": "gpu_hist",
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
        fit_params = {
            "verbose": True,
        }
        model = xgb.XGBClassifier(**model_params)
        eval_metric = accuracy_score
    else:
        raise NotImplementedError

    dm.setup(stage=None)
    x_train, y_train = utils.dm_to_numpy(dm.train_dataloader())
    x_valid, y_valid = utils.dm_to_numpy(dm.val_dataloader())
    x_test, y_test = utils.dm_to_numpy(dm.test_dataloader())
    model.fit(
        x_train,
        y_train,
        eval_set=[(x_train, y_train), (x_valid, y_valid)],
        **fit_params,
    )
    y_test_preds = model.predict(x_test)
    test_score = eval_metric(y_test, y_test_preds)
    print(f"test score: {test_score}")


if __name__ == "__main__":
    fire.Fire(main)
