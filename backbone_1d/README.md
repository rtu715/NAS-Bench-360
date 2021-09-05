# Experiments on Wide ResNet - 1D

With a Determined cluster set up, you can edit experimental scripts in scripts/ and run
experiments using the following commands:

`det experiment create scripts/[experiment name].yaml .`

For running a local test simply run with: 

`det experiment create scripts/[experiment name].yaml . --local --test`


`[dataset].yaml` - Without tuning (fixed hyperparameters)

`*tuning.yaml` -  Hyperparameter tuning 

`*retrain.yaml` - Retrain with searched hyperparameters


