from deepCR import deepCR
from deepCR import evaluate
import matplotlib.pyplot as plt
import numpy as np
import os


base_dir = '.'
mdl = deepCR(mask='2021-09-10_mymodel_epoch50.pth', hidden=32)

test_dirs = np.load(os.path.join(base_dir, 'test_dirs.npy'), allow_pickle=True)
tpr, fpr = evaluate.roc(mdl, test_dirs=test_dirs, thresholds=np.array([0.5]), seed=2)
print(tpr, fpr)
auroc = 0.5 - fpr/2/100 + tpr/2/100
print(auroc)
print(1-auroc)
plt.plot(fpr, tpr)
plt.show()
