# NAS-Bench-360

Some information Here

| Tasks               | Number of Samples | Data Split(train/val/test) | Task Type | Applications           | License  |
|---------------------|-------------------|----------------------------|-----------|------------------------|----------|
| CIFAR-100           | 60,000            | 40,000/10,000/10,000       | Point     | Computer Vision        | CC BY-SA |
| Permuted CIFAR-100  | 60,000            | 40,000/10,000/10,000       | Point     | Transformed Vision     | CC BY-SA |
| Spherical CIFAR-100 | 60,000            | 40,000/10,000/10,000       | Point     | Omnidirectional Vision | CC BY-SA |
| Ninapro DB5         | 3,916             | 2,638/659/659              | Point     | Medical Imaging        | CC BY-ND |
| Darcy Flow          | 1,100             | 900/100/100                | Grid      | PDE Solver             | MIT      |
| PsiCov + DeepCov    | 3,456 + 150       | 3356/100/150               | Grid      | Protein Folding        | GPL      |
*validation data can be split through index slicing 

## Metadata and Download Links
Metadata is in the form of Data Nutrition Labels
### CIFAR-100 <br /> 
[Download](https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz) 
(161 MB, our code loads from torchvision)

### Permuted CIFAR-100 <br /> 
[Metadata](permuted.pdf) <br/>
[Download](https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz)
(161 MB, we generate permutated images in the code directly)

### Spherical CIFAR-100 <br /> 
[Metadata](spherical.pdf) <br/>
[Download](https://pde-xd.s3.amazonaws.com/spherical/s2_cifar100.gz) (272 MB)

### Processed Ninapro DB5  <br />
Download Links (~30 MB total): <br />
[Train Data](https://pde-xd.s3.amazonaws.com/ninapro/ninapro_train.npy), 
[Train Labels](https://pde-xd.s3.amazonaws.com/ninapro/label_train.npy); 
[Validation Data](https://pde-xd.s3.amazonaws.com/ninapro/ninapro_val.npy),
[Validation Labels](https://pde-xd.s3.amazonaws.com/ninapro/label_val.npy); 
[Test Data](https://pde-xd.s3.amazonaws.com/ninapro/ninapro_test.npy),
[Test Labels](https://pde-xd.s3.amazonaws.com/ninapro/label_test.npy)

### Darcy Flow <br />
[Download Train + Validation Data](https://pde-xd.s3.amazonaws.com/piececonst_r421_N1024_smooth1.mat) (1.6 GB) <br/>
[Download Test Data](https://pde-xd.s3.amazonaws.com/piececonst_r421_N1024_smooth2.mat) (1.6 GB)

### DeepCov + PsiCov (Protein Folding) <br />
[Download](https://pde-xd.s3.amazonaws.com/protein.zip) (1.1 GB)

## Reading data 

Despite data is stored using various formats, we provide sample python scripts for loading
all datasets into pytorch tensors. From there, you can use them for training or maybe visualize
the images. 

Point task script: [point.py](point.py)

Grid task scripts: [grid.py](grid.py), [utils](utils_grid.py), [protein_generator](protein_gen.py),
[protein_io](protein_io.py)

## Benchmark Results 

## Reproducibility Checklist 