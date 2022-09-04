# AutoML for Image Semantic Segmentation
Currently this repo contains the only working open-source implementation of [Auto-Deeplab](https://arxiv.org/abs/1901.02985) which, by the way **out-performs** that of the original paper. 

## Using our docker container
`docker pull renbotu/xd-nas:determined1.8`
## Training Procedure

**All together there are 3 stages:**

1. Architecture Search - Here you will train one large relaxed architecture that is meant to represent many discreet smaller architectures woven together.

2. Decode - Once you've finished the architecture search, load your large relaxed architecture and decode it to find your optimal architecture.

3. Re-train - Once you have a decoded and poses a final description of your optimal model, use it to build and train your new optimal model

<br/><br/>

 
 ## Architecture Search

***Begin Architecture Search***

**Start Training**

```
CUDA_VISIBLE_DEVICES=0 python train_autodeeplab.py --dataset darcyflow
```

**Resume Training**

```
CUDA_VISIBLE_DEVICES=0 python train_autodeeplab.py --dataset darctflow --resume /AutoDeeplabpath/checkpoint.pth.tar
```

## Re-train

***Now that you're done training the search algorithm, it's time to decode the search space and find your new optimal architecture. 
After that just build your new model and begin training it***


**Load and Decode**
for the resume flag, locate your search code output directory
```
CUDA_VISIBLE_DEVICES=0 python decode_autodeeplab.py --dataset darcyflow --resume /AutoDeeplabpath/checkpoint.pth.tar
```

## Retrain

**Train without distributed**
```
python train.py --dataset darcyflow --exp {/path/to/decoded_arch}
```

**Train with distributed**
```
CUDA_VISIBLE_DEVICES=0,1,2,···,n python -m torch.distributed.launch --nproc_per_node=n train_distributed.py  
```


## References
[1] : [Auto-DeepLab: Hierarchical Neural Architecture Search for Semantic Image Segmentation](https://arxiv.org/abs/1901.02985)

[2] : [Thanks for jfzhang's deeplab v3+ implemention of pytorch](https://github.com/jfzhang95/pytorch-deeplab-xception)

[3] : [Thanks for MenghaoGuo's autodeeplab model implemention](https://github.com/MenghaoGuo/AutoDeeplab)

[4] : [Thanks for CoinCheung's deeplab v3+ implemention of pytorch](https://github.com/CoinCheung/DeepLab-v3-plus-cityscapes)

[5] : [Thanks for chenxi's deeplab v3 implemention of pytorch](https://github.com/chenxi116/DeepLabv3.pytorch)

