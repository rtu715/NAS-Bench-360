# Geometry-Aware Exponential Algorithms for Neural Architecture Search (GAEA)
This example implements the NAS method introduced by [Li et al.](https://arxiv.org/abs/2004.07802) called GAEA for geometry-aware neural architecture search (check out the paper for more details about the NAS algorithm).  GAEA is state-of-the-art among NAS method on the [DARTS search space](https://arxiv.org/abs/1806.09055).  You can replicate the results in the paper and try GAEA for your own data using the code provided here.

## Architecture Search
The training routine is based on that used by [PC-DARTS](https://github.com/yuhuixu1993/PC-DARTS/blob/master/train_imagenet.py). 

### To Run
To run the example, simply run the following command from the `search` directory:
` det experiment create [dataset]_search.yaml .`

After the architecture search stage is complete, you can evaluate the architecture by copying the resulting genotype from the log to `model_def.py`.  

## Architecture Evaluation
Run the following command from the `eval` directory:
` det experiment create [dataset]_eval.yaml .`