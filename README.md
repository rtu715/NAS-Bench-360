# NAS-Bench-360

## Resources 
Homepage / dataset downloads: [here](https://rtu715.github.io/NAS-Bench-360/)

Datasets in the benchmark:
- CIFAR-100 (Image classification)
- Spherical CIFAR-100 (Transformed image classification)
- Ninapro DB5 (Hand-gesture classification)
- Darcy Flow (Partial differential equation solver)
- PSICOV (Protein sequence distance prediction)

## Prerequisites 
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

## Experiment Reproduction
We provide pytorch implementations for two state-of-the-art NAS algorithms: GAEA PC-DARTS ([paper link](https://arxiv.org/pdf/2004.07802.pdf))
and DenseNAS ([paper link](https://arxiv.org/abs/1906.09607)), 
which can be found inside each folder with the associated name, i.e. "determined-darts/" for GAEA PC-DARTS 
and "determined-densenas/" for DenseNAS.

We also implement procedure for running and tuning hyperparameters of the backbone architecture Wide ResNet ([paper link]( http://arxiv.org/abs/1605.07146)), in "backbone/".

To modify the random seed for each experiment, modify the number under 

`reproducibility: experiment_seed: ` for each script









