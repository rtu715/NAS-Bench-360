[![Documentation Status](https://readthedocs.org/projects/amber-automl/badge/?version=latest)](https://amber-automl.readthedocs.io/en/latest/?badge=latest)
[![Coverage](https://raw.githubusercontent.com/zj-zhang/AMBER/master/amber/tests/coverage-badge.svg)]()
[![Latest Release](https://img.shields.io/github/release/zj-zhang/AMBER.svg?label=Release)](https://github.com/zj-zhang/AMBER/releases/latest)
[![Downloads](https://pepy.tech/badge/amber-automl)](https://pepy.tech/project/amber-automl)
[![DOI](https://zenodo.org/badge/260604309.svg)](https://zenodo.org/badge/latestdoi/260604309)
<!-- 
[![PyPI Install](https://img.shields.io/pypi/dm/amber-automl.svg?label=PyPI%20Installs)](https://pypi.org/project/amber-automl/)
[![Github All Releases](https://img.shields.io/github/downloads/zj-zhang/AMBER/total.svg?label=Download)](https://github.com/zj-zhang/AMBER/releases)
-->

![logo](docs/source/_static/img/amber-logo.png)

---

**Automated Modeling for Biological Evidence-based Research**


AMBER is a toolkit for designing high-performance neural network models automatically in
Genomics and Bioinformatics.

The overview, tutorials, API documentation can be found at:
https://amber-automl.readthedocs.io/en/latest/

## Please use our docker image for the environment
`docker pull renbotu/nb360:tensorflow`

## Reinstall AMBER from local directory (out REPO)

`pip install /path/to/AMBER`

## Experiment scripts for architecture search 
To run ECG, satellite, or DeepSEA:
`python examples/amber_{taskname}`

## For evaluation 
Modify examples/eval.py to locate the directory for storing search outputs
`python examples/eval.py`

## Contact

If you find AMBER useful in your research, please cite the following paper:

Zhang Z, Park CY, Theesfeld CL, Troyanskaya OG. An automated framework for efficiently designing deep convolutional neural networks in genomics. Nature Machine Intelligence. 2021 Mar 15:1-9. [Paper](https://www.nature.com/articles/s42256-021-00316-z) [Preprint](https://www.biorxiv.org/content/10.1101/2020.08.18.251561v1.full)

[Back to Top](#sec0)







