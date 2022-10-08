from deepCR import train
import numpy as np
import os

base_dir = '../../autodeeplab/'

train_dirs = np.load(os.path.join(base_dir, "train_dirs.npy"), allow_pickle=True)
test_dirs = np.load(os.path.join(base_dir, "test_dirs.npy"), allow_pickle=True)

trainer = train(
    train_dirs,
    test_dirs,
    ignore=0,
    sky=1,
    aug_sky=[-0.9, 3],
    name="mymodels2",
    epoch=50,
    save_after=20,
    plot_every=10,
    use_tqdm=True,
)
trainer.train()
filename = trainer.save()
