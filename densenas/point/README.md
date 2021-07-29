
## Search

We perform search using the ResNet-based search space. 

Run `det experiment create scripts/search/[dataset].yaml .`
## Train

Copy the searched architecture from experiment logs to the evaluation scripts `scripts/eval/[dataset].yaml`

Run `det experiment create scripts/eval/[dataset].yaml .`