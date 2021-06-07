# NAS-Bench-360

Dataset downloads are available [here](https://rtu715.github.io/NAS-Bench-360/)

We use the open-source [Determined](https://docs.determined.ai/latest/how-to/installation/aws.html?highlight=det%20deploy) 
software to implement experiment code, and we provide NAS implementations for GAEA PC-DARTS and DenseNAS, 
which can be found inside each folder with the associated name, i.e. "determined-darts/" for GAEA PC-DARTS 
and "determined-densenas/" for DenseNAS.

Code for running and tuning hyperparameters of the backbone architecture, Wide ResNet, is located in "backbone/".

To modify the random seed for each experiment, modify the number under 

`reproducibility: experiment_seed: ` for each script









