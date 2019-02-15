# QuickNat - Pytorch implementation

Tool: QuickNAT: Segmenting MRI Neuroanatomy in 20 seconds
-----------------------------------------------------------

Authors: Abhijit Guha Roy, Sailesh Conjeti, Nassir Navab and Christian Wachinger

The code for training, as well as the Trained Models are provided here.

Deployment of existing off-the-shelf Model to segment any MRI scans is just by running RunFile.

Let us know if you face any problems running the code by posting in Issues.

If you use this code please cite:

Guha Roy, A., Conjeti, S., Navab, N., and Wachinger, C. 2018. QuickNAT: Segmenting MRI Neuroanatomy in 20 seconds. Accepted for publication at **NeuroImage**.
 
 Link to paper: https://arxiv.org/abs/1801.04161 
 
 Enjoy!!! :)
 

## Getting Started

### Pre-requisites

You need to have following in order for this library to work as expected
1. Python >= 3.5
2. Pip >= 19.0
2. Pytorch >= 1.0.0
3. Numpy >= 1.14.0
4. Squeeze and Excitation (https://github.com/ai-med/squeeze_and_excitation, Follow this link to know more about installation and usage)

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
