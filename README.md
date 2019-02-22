# QuickNat - Pytorch implementation

Tool: QuickNAT: Segmenting MRI Neuroanatomy in 20 seconds
-----------------------------------------------------------

Authors: Abhijit Guha Roy, Sailesh Conjeti, Nassir Navab and Christian Wachinger

The code for training, as well as the Trained Models are provided here.

Deployment of existing off-the-shelf Model to segment any MRI scans is just by running RunFile.

Let us know if you face any problems running the code by posting in Issues.

If you use this code please cite:

Guha Roy, A., Conjeti, S., Navab, N., and Wachinger, C. 2018. QuickNAT: A Fully Convolutional Network for Quick and Accurate Segmentation of Neuroanatomy. Accepted for publication at **NeuroImage**, https://arxiv.org/abs/1801.04161. 
 
Online demo: http://quicknat.ai-med.de 
 
 Enjoy!!! :)
 

## Getting Started

### Pre-requisites

You need to have following in order for this library to work as expected
1. python >= 3.5
2. pip >= 19.0
3. pytorch >= 1.0.0
4. numpy >= 1.14.0
5. nn-common-modules >=1.0 (https://github.com/ai-med/nn-common-modules, A collection of commonly used code modules in deep learning. Follow this link to know more about installation and usage)
6. Squeeze and Excitation >=1.0 (https://github.com/ai-med/squeeze_and_excitation, Follow this link to know more about installation and usage)

### Training your model

```
python run.py --mode=train
```

### Evaluating your model

```
python run.py --mode=eval
```

## Code Authors

* **Shayan Ahmad Siddiqui**  - [shayansiddiqui](https://github.com/shayansiddiqui)
* **Abhijit Guha Roy**  - [abhi4ssj](https://github.com/abhi4ssj)


## Help us improve
Let us know if you face any issues. You are always welcome to report new issues and bugs and also suggest further improvements. And if you like our work hit that start button on top. Enjoy :)
