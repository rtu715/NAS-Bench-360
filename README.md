# NAS-Bench-360

This codebase reproduces various empirical evaluations on [NAS-Bench-360](https://nb360.ml.cmu.edu/), a benchmark for evaluating neural architecture search on diverse tasks, that can be found in the associated [paper](https://arxiv.org/abs/2110.05668).

## Resources 
<b> Oct 2022: Please use this link for all dataset and precompute downloads as we are transitioning our AWS resources to the Open Data Sponsorship program:

[Shared Drive](https://drive.google.com/drive/folders/1OOp6UrCiP79mTIkCrUdjjHWDo8K0bJUw)  

</b>

Datasets in the benchmark with download links (<b> old links, use the shared drive above! </b>):
- CIFAR-100 (Image classification)
- [Spherical CIFAR-100 (Transformed image classification)](https://pde-xd.s3.amazonaws.com/spherical/s2_cifar100.gz) (272 MB)
- [Ninapro DB5 (Hand-gesture classification)](https://pde-xd.s3.amazonaws.com/ninapro/ninapro_train.npy)(15 MB)
- [Darcy Flow (Partial differential equation solver)](https://pde-xd.s3.amazonaws.com/piececonst_r421_N1024_smooth1.mat) (1.6 GB) 
- [PSICOV (Protein sequence distance prediction)](https://pde-xd.s3.amazonaws.com/protein.zip) (1.1 GB)
- [FSD50k (Sound event classification)](https://pde-xd.s3.amazonaws.com/audio/audio.zip) (24 GB)
- [Cosmic (Cosmic ray identification and replacement)](https://pde-xd.s3.amazonaws.com/cosmic/deepCR.ACS-WFC.train.tar) (6.5 GB)
- [ECG (Cardiac anomaly detection)](https://pde-xd.s3.amazonaws.com/ECG/challenge2017.pkl)(150 MB)
- [Satellite (Earth monitoring through satellite imagery)](https://pde-xd.s3.amazonaws.com/satellite/satellite_train.npy) (322 MB)
- [DeepSEA (identifying chromatin features from DNA sequences)](https://pde-xd.s3.amazonaws.com/deepsea/deepsea_filtered.npz)(860 MB)

Precomputed evaluation benchmark files on the NB201 search space (following NATS-Bench):
- [NinaPro DB5](https://drive.google.com/file/d/1y_Y3TbIE5rVhJ42hwIq6alUeMqlFr-bv/view?usp=sharing)(84 MB)
- [Darcy Flow](https://drive.google.com/file/d/1PXYdAjtuBz9aR4MMzbjYbVzojTQrfliU/view?usp=sharing) (85 MB)


For full outputs (include training logs and all weights and checkpoints), please contact the administrators. They are of size ~40 GB.
<!---
- [NinaPro DB5](https://pde-xd.s3.amazonaws.com/ninapro_precompute.zip)(46 GB)
- [Darcy Flow](https://pde-xd.s3.amazonaws.com/darcyflow_precompute.zip) (35.4 GB)
-->

## Prerequisites for main NAS experiments
We use the open-source [Determined](https://docs.determined.ai/latest/how-to/installation/aws.html?highlight=det%20deploy) 
software to implement experiment code. 

Installing determined: `pip install determined`

A [master instance](https://docs.determined.ai/latest/how-to/installation/deploy.html) is required:
- for local deployment (need to [install docker](https://docs.determined.ai/latest/how-to/installation/requirements.html#install-docker)):
  - to start the master: `det deploy local cluster-up`
  - access the WebUI at `http://localhost:8080`
  - to shut down: `det deploy local cluster-down`
    
- for AWS deployment (preferred):
  - [install AWS CLI](https://docs.aws.amazon.com/cli/latest/userguide/cli-chap-install.html)
  - Run `aws configure` and find AWS EC2 keypair name
  - to start the master: `det deploy aws up --cluster-id CLUSTER_ID --keypair KEYPAIR_NAME`
  - access the WebUI at `{ec2-instance-uri}:8080`
  - to shut down: `det deploy aws down --cluster-id CLUSTER_ID`
    
For an end-to-end example of running experiments with determined, you can refer to this [video](https://www.youtube.com/watch?v=htObOwwnhQk&t=394s).

When running experiments, a docker image is automatically pulled from docker hub which contains all required python packages
, i.e. you don't need to install them yourself, and it ensures reproducibility. 

## Main NAS Experiments Reproduction
We provide pytorch implementations for two state-of-the-art NAS algorithms: GAEA PC-DARTS ([paper link](https://arxiv.org/pdf/2004.07802.pdf))
and DenseNAS ([paper link](https://arxiv.org/abs/1906.09607)), 
which can be found inside each folder with the associated name, i.e. `darts/` for GAEA PC-DARTS 
and `densenas/` for DenseNAS.

To run these algorithms on 1D tasks, we've adapted their search spaces whose experiments are provided in `darts_1d/` for GAEA PC-DARTS (1D) and `densenas_1d/` for DenseNAS(1D). 

Two task-specific NAS methods are implemented: Auto-DeepLab for dense prediction tasks in `autodeeplab/` and AMBER for 1D prediction tasks in `AMBER/`.

We also implement procedure for running and tuning hyperparameters of the backbone architecture Wide ResNet ([paper link](http://arxiv.org/abs/1605.07146)), in `backbone/`. The 1D-customized Wide ResNet is in `backbone_1d/`.

To modify the random seed for each experiment, modify the number under 

`reproducibility: experiment_seed: ` for each script

# Additional Baseline Experiments
We also evaluate the performance of non-NAS baselines for comparison:
- Expert architectures for each dataset: see `expert`.
- [Perceiver-IO](https://arxiv.org/abs/2107.14795): see `perceiver-io`.
- [XGBoost](https://arxiv.org/abs/1603.02754?context=cs#): see `xgboost`.

# Precomputed results on NinaPro and DarcyFlow
- See the `precompute` directory for NAS algorithms from NATS-Bench and reproduction of the precomputed benchmark. 


## Baselines
Performance of NAS and baselines across NAS-Bench-360. Methods are divided into efficient methods (e.g. DenseNAS and fixed WRN) that take 1-10 GPU-hours, more expensive methods (e.g. DARTS and tuned WRN) that take 10-100+ GPU-hours, and specialized methods (Auto-DL and AMBER). All results are averages of three random seeds, and lower is better for all metrics.
![alt text](https://github.com/rtu715/NAS-Bench-360/blob/main/images/baselines.png)

## Citation
If you find this project helpful, please consider citing our paper:
```bibtex
@inproceedings{
  tu2022nasbench,
  title={{NAS}-Bench-360: Benchmarking Neural Architecture Search on Diverse Tasks},
  author={Renbo Tu and Nicholas Roberts and Mikhail Khodak and Junhong Shen and Frederic Sala and Ameet Talwalkar},
  booktitle={Thirty-sixth Conference on Neural Information Processing Systems Datasets and Benchmarks Track},
  year={2022},
  url={https://openreview.net/forum?id=xUXTbq6gWsB}
}
```
Thanks!
