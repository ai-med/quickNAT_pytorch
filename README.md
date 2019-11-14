# QuickNat and Bayesian QuickNAT - Pytorch implementation

A fully convolutional network for quick and accurate segmentation of neuroanatomy and Quality control of structure-wise segmentations
-----------------------------------------------------------

The code for training, as well as the Trained Models are provided here.

Deployment of existing off-the-shelf Model to segment any MRI scans is just by running RunFile.

Let us know if you face any problems running the code by posting in Issues.

If you use this code please cite the following papers:
```
@article{roy2019quicknat,
  title={QuickNAT: A fully convolutional network for quick and accurate segmentation of neuroanatomy},
  author={Roy, Abhijit Guha and Conjeti, Sailesh and Navab, Nassir and Wachinger, Christian and Alzheimer's Disease Neuroimaging Initiative and others},
  journal={NeuroImage},
  volume={186},
  pages={713--727},
  year={2019},
  publisher={Elsevier}
}

@article{roy2019bayesian,
  title={Bayesian QuickNAT: Model uncertainty in deep whole-brain segmentation for structure-wise quality control},
  author={Roy, Abhijit Guha and Conjeti, Sailesh and Navab, Nassir and Wachinger, Christian and Alzheimer's Disease Neuroimaging Initiative and others},
  journal={NeuroImage},
  volume={195},
  pages={11--22},
  year={2019},
  publisher={Elsevier}
}
```
Online demo for trying out: http://quicknat.ai-med.de 

Link to arxiv versions of the papers:
* [QuickNAT](https://arxiv.org/abs/1801.04161)
* [Bayesian QuickNAT](https://arxiv.org/abs/1811.09800)

Enjoy!!! :)
 

## Getting Started

### Pre-requisites
Please install the required packages for smooth functioning of the tool by running
```
pip install -r requirements.txt
```

### Training your model

```
python run.py --mode=train
```

### Evaluating your model

```
python run.py --mode=eval
```

## Evaluating the model in bulk

Execute the following command for deploying on large datasets:
```
python run.py --mode=eval_bulk
```
This saves the segmentation files at nifti files in the destination folder. Also in the folder, a '.csv' file is generated which provides the volume estimates of the brain structures with subject ids for all the processed volumes.

Also uncertainty flag is set, another two '.csv' files are created with structure-wise uncertainty (CVs and IoU) for quality control of the segmentations. Please refer to the "Bayesian QuickNAT" paper for details.

**Pre-processing**: Before deploying our model you need to standardize the MRI scans. Use the following command from FreeSurfer 
```
mri_convert --conform <input_volume.nii> <out_volume.nii>
```
The above command standardizes the alignment for QuickNAT, re-samples to isotrophic resolution (256x256x256) with some contrast enhamcement. It takes about one second per volume.

You need to modify the following entries in 'settings_eval.ini' file in the repo.

* **device**: CPU or ID of GPU (0 or 1) you want to excecute your code.
* **coronal_model_path**: It is by default set to "saved_models/finetuned_alldata_coronal.pth.tar" which is our final model. You may also use "saved_models/IXI_fsNet_coronal.pth.tar" which is our pre-trained model.
* **axial_model_path**: Similar to above. It is only used for view_aggregation stage.
* **data_dir**: Absolute path to the data directory where input volumes are present.
* **directory_struct**: Valid options are "FS" or "Linear". If you input data directory is similar to FreeSurfer, i.e. **data_dir**/<Data_id>/mri/orig.mgz then use "FS". If the entries are **data_dir**/<Data_id> use "Linear".
* **volumes_txt_file**: Path to the '.txt' file where the data_ID names are stored. If **directory_struct** is "FS" the entries should be only the folder names, whereas if it is "Linear" the entry name should be the file names with the file extensions.
* **batch_size**: Set this according the capacity of your GPU RAM.
* **save_predictions_dir**: Indicate the absolute path where you want to save the segmentation outputs along with the '.csv' files for volume and uncertainty estimates.
* **view_agg**: Valid options are "True" or "False". When "False", it uses coronal network by default.
* **estimate_uncertainty**: Valid options are "True" or "False". Indicates if you want to estimate the structure-wise uncertainty for segmentation Quality control. Refer to "Bayesian QuickNAT" paper for more details.
* **mc_samples**: Active only if **estimate_uncertainty** flag is "True". Indicates the number of Monte-Carlo samples used for uncertainty estimation. 
* **labels**: List of label names used in '.csv' file. Do not change this unless you change the model.

 


## Code Authors

* **Shayan Ahmad Siddiqui**  - [shayansiddiqui](https://github.com/shayansiddiqui)
* **Abhijit Guha Roy**  - [abhi4ssj](https://github.com/abhi4ssj)


## Help us improve
Let us know if you face any issues. You are always welcome to report new issues and bugs and also suggest further improvements. And if you like our work hit that start button on top. Enjoy :)