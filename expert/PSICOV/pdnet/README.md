# PDNET
## A fully open-source framework for deep learning protein real-valued distances

As deep learning algorithms drive the progress in protein structure prediction, a lot remains to be studied at this merging superhighway of deep learning and protein structure prediction. Recent findings show that inter-residue distance prediction, a more granular version of the well-known contact prediction problem, is a key to predicting accurate models. However, deep learning methods that predict these distances are still in the early stages of their development. To advance these methods and develop other novel methods, a need exists for a small and representative dataset packaged for faster development and testing. In this work, we introduce protein distance net (PDNET), a framework that consists of one such representative dataset along with the scripts for training and testing deep learning methods. The framework also includes all the scripts that were used to curate the dataset, and generate the input features and distance maps. Deep learning models can also be trained and tested in a web browser using free platforms such as Google Colab. We discuss how the PDNET framework can be used to predict contacts, distance intervals, and real-valued distances.

## Full dataset
http://deep.cs.umsl.edu/pdnet/  

## Distance prediction compared with the image depth prediction problem
![](./pdnet.png)
(Figure above) Comparison of the protein inter-residue distance prediction problem with the 'depth prediction from single
image problem' in computer vision. In both problems the input to the deep learning model is a volume and the
output is a 2D matrix. The depth predictions for this specific image (top right corner) were obtained by running the
pretrained FCRN method.

## Presentation@Youtube
[https://www.youtube.com/watch?v=uAIuA1O7iE8](https://www.youtube.com/watch?v=uAIuA1O7iE8)

## Where to start?
### In a server without notebook
Download dataset:
```bash
wget http://deep.cs.umsl.edu/pdnet/train-data.tar.gz
tar zxvf train-data.tar.gz
```
Start training:
```
python3 train.py -h
```
### In Google Colab
Open the `pdnet_distance.ipynb` file inside the `notebooks` folder in [Google Colab](https://colab.research.google.com/) and select a GPU runtime environment. If you are new to Google Colab, please watch [this](https://www.youtube.com/watch?v=PVsS9WtwVB8).

## Reference
"A fully open-source framework for deep learning protein real-valued distances", Scientific Reports
DOI: [https://www.nature.com/articles/s41598-020-70181-0](https://www.nature.com/articles/s41598-020-70181-0)

## Contact
Badri Adhikari  
https://badriadhikari.github.io/  
