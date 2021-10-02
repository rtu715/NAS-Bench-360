# Bioinfor DeepSEA

DeepSEA is a model to predicting effects of noncoding variants.
This is implemented by tensorflow-2.0 again.


## DeepSEA

### Model Architecture
CNN + Pool + CNN + Pool + CNN + Dense + Dense

### Loss Function
Binary Cross Entropy

### Optimization Method
SGD with Momentum (momentum = 0.9)

## USAGE

### Requirement
We run the code on Ubuntu 18.04 LTS with a GTX 1080ti GPU.

[Python](<https://www.python.org>) (3.7.3) | [Tensorflow](<https://tensorflow.google.cn/install>) (2.0.0)
| [CUDA](<https://developer.nvidia.com/cuda-toolkit-archive>) (10.0) | [cuDNN](<https://developer.nvidia.com/cudnn>) (7.6.0)

### Data
You need to first download the training, validation, and testing sets from DeepSEA. You can download the datasets from 
[here](<http://deepsea.princeton.edu/media/code/deepsea_train_bundle.v0.9.tar.gz>). After you have extracted the
contents of the tar.gz file, move the 3 .mat files into the **`./data/`** folder.

### Model
The model that trained by myself is available in BAIDU Net Disk [here](https://pan.baidu.com/s/1tfYvDoO6Xvt7v7y70nDsXg)


### Preprocess
Because of my RAM limited, I firstly transform the train.mat file to .tfrecord files.
```
python preprocess.py
```

### Training
Then you can train the model initially.
```
CUDA_VISIBLE_DEVICES=0 python main.py -e train
```

### Test
When you have trained successfully, you can evaluate the model.
```
CUDA_VISIBLE_DEVICES=0 python main.py -e test
```

## RESULT
You can get my result in the **`./result/`** directory.

### Loss Curve
![model loss](./result/model_loss.jpg)

### Metric
We use two metric to evaluate the model. (AUROC, AUPR)

-|DNase|TFBinding|HistoneMark|All
:-:|:-:|:-:|:-:|:-:
AUROC|0.8925|0.9071|0.8270|0.8960
AUPR|0.3809|0.2400|0.3277|0.2691


## REFERENCE
> [Predicting effects of noncoding variants with deep learning-based sequence model](<https://www.nature.com/articles/nmeth.3547>) | [Github](<https://github.com/jisraeli/DeepSEA>)