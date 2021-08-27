---
schemadotorg:
 "@context": http://schema.org/
 "@type": CreativeWork
 about: "This is a NAS benchmark introducing novel tasks"
 audience:
   - "@type": Audience
     name: WebMaster
 genre: "Benchmark"
 name: "NAS-Bench-360"
 author: ["Renbo Tu"]
 contributor:
   - "@type": Person
     name: "Mikhail Khodak"
 description: "a NAS benchmark for diverse tasks, i.e. instead of nasbench101, 201, 301 on a bunch of vision tasks we come up with a suite of many interesting tasks from different fields, 
     run different nas algos/search spaces on them."
 keywords: ["schemaorg", "TeSS"]
 license: MIT
---

# NAS-Bench-360


| Tasks               | Number of Samples | Data Split(train/val/test) | Task Type | Applications           | License  |
|---------------------|-------------------|----------------------------|-----------|------------------------|----------|
| CIFAR-100           | 60K               | 40,000/10,000/10,000       | Point     | Computer Vision        | CC BY 4.0|
| Spherical CIFAR-100 | 60K               | 40,000/10,000/10,000       | Point     | Omnidirectional Vision | CC BY-SA |
| Ninapro DB5         | 3,916             | 2,638/659/659              | Point     | Prosthetics Control    | CC BY-ND |
| FSD50k              | 51K               | 37K/4K/10K                 | Point     | Audio Classification   | CC BY 4.0 |
| Darcy Flow          | 1,100             | 900/100/100                | Dense     | PDE Solver             | MIT      |
| PSICOV + DeepCov    | 3,456 + 150       | 3,356/100/150              | Dense     | Protein Folding        | GPL      |
| Cosmic              | 5,250             | 4,347/483/420              | Dense     | Astronomy Imaging      | Open Data|
| ECG                 | 330K              | 260K/33K/33K               | 1D Point  | Medical Diagnostics    | ODC-BY 1.0|
| Satellite           | 1M                | 800K/100K/100K             | 1D Point  | Earth Minitoring       | GPL      |
| DeepSEA             | 250K              | 71K/2.5K/150K              | 1D Point  | Genetic Prediction     | CC BY 4.0|

*validation data can be split through index slicing 

## Metadata and Download Links
Metadata is in the form of Data Nutrition Labels
### CIFAR-100 <br /> 
[Download](https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz) 
(161 MB, our code loads from torchvision)

<!--
### Permuted CIFAR-100 <br /> 
[Metadata](permuted.pdf) <br/>
[Download](https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz)
(161 MB, we generate permutated images in the code directly)
-->

### Spherical CIFAR-100 <br /> 
[Metadata](spherical.pdf) <br/>
[Download](https://pde-xd.s3.amazonaws.com/spherical/s2_cifar100.gz) (272 MB)

### Processed Ninapro DB5 (sEMG)  <br />
Download Links (~30 MB total): <br />
[Train Data](https://pde-xd.s3.amazonaws.com/ninapro/ninapro_train.npy), 
[Train Labels](https://pde-xd.s3.amazonaws.com/ninapro/label_train.npy); 
[Validation Data](https://pde-xd.s3.amazonaws.com/ninapro/ninapro_val.npy),
[Validation Labels](https://pde-xd.s3.amazonaws.com/ninapro/label_val.npy); 
[Test Data](https://pde-xd.s3.amazonaws.com/ninapro/ninapro_test.npy),
[Test Labels](https://pde-xd.s3.amazonaws.com/ninapro/label_test.npy)

### FSD50K <br />
[Download](https://pde-xd.s3.amazonaws.com/audio/audio.zip) (24 GB)

### Darcy Flow (PDE) <br />
[Download Train + Validation Data](https://pde-xd.s3.amazonaws.com/piececonst_r421_N1024_smooth1.mat) (1.6 GB) <br/>
[Download Test Data](https://pde-xd.s3.amazonaws.com/piececonst_r421_N1024_smooth2.mat) (1.6 GB)

### DeepCov + PSICOV (Protein Folding) <br />
[Download](https://pde-xd.s3.amazonaws.com/protein.zip) (1.1 GB)

### Cosmic <br />
[Download Train](https://pde-xd.s3.amazonaws.com/cosmic/deepCR.ACS-WFC.train.tar) (6.5 GB)
[Download Test](https://pde-xd.s3.amazonaws.com/cosmic/deepCR.ACS-WFC.test.tar) (2.0 GB)

### ECG <br />
[Download](https://pde-xd.s3.amazonaws.com/ECG/challenge2017.pkl)(150 MB)

### Satellite <br />
[Download Train](https://pde-xd.s3.amazonaws.com/satellite/satellite_train.npy) (322 MB)
[Download Test](https://pde-xd.s3.amazonaws.com/satellite/satellite_test.npy) (35 MB)

### DeepSEA <br />
[Download](https://pde-xd.s3.amazonaws.com/deepsea/deepsea_filtered.npz)(860 MB)


## Reading data 

Despite data is stored using various formats, we provide sample python scripts for loading
all datasets into pytorch tensors. From there, you can use them for training or maybe visualize
the images. Loading 1D tasks is similar to loading point tasks. 

Point task script: [point.py](point.py)

Dense task scripts: [dense.py](dense.py), [utils](utils_grid.py), [protein_generator](protein_gen.py),
[protein_io](protein_io.py)

<!--
## Benchmark Results 

| Tasks               | GAEA PC-DARTS  | DenseNAS     | Backbone + HPO | Human-designed architecture |
|---------------------|----------------|--------------|----------------|-----------------------------|
| CIFAR-100           | 75.81 ± 2.12   | 72.56 ± 0.65 | 75.11 ± 0.23   | 82.83                       |
| Spherical CIFAR-100 | 47.10 ± 4.08   | 27.01 ± 0.95 | 21.55 ± 0.60   | 35.58                       |
| Ninapro             | 88.57 ± 0.61   | 89.83 ± 1.31 | 93.12 ± 0.40   | 68.98                       |
| Darcy Flow          | 0.056 ± 0.012  | 0.10 ± 0.010 | 0.041 ± 0.0012 | 0.0065                      |
| PSICOV              | 2.80 ± 0.057   | 3.84 ± 0.15  | 5.71 ± 0.15    | 3.50                        |
-->

## License

The code is open-source and released under the [MIT](https://en.wikipedia.org/wiki/MIT_License) LICENSE