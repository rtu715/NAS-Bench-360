
## Search

We perform search using the ResNet-based (1D) search space. 

Run `det experiment create scripts/search/[dataset].yaml .`

To perform Random Search, you would need a searched architecture (read from experiment logs) from the regular method and modify the "target_arch" field in `scripts/search/[dataset]_random.yaml`. Then run `det experiment create scripts/search/[dataset]_random.yaml .`

## Train

Copy the searched architecture from experiment logs to the evaluation scripts `scripts/eval/[dataset].yaml`

Run `det experiment create scripts/eval/[dataset].yaml .`