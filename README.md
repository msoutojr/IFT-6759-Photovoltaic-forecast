# IFT-6759 - Projets avancés en apprentissage automatique
## Project: Photovoltaic Power and Solar Radiation Forecasting

Project developed during the course IFT-6759 at the University of Montreal, in the winter semester 2022.

## Team:
* Marcos A. A. Souto Jr.
* Kodjovi Adabra
* Niranjan Niranjan
* Rui Ze Ma

## Project Goal

The main theme of the project is photovoltaic energy generation forecast. The objective is to reproduce and extend the methodology to improve the results found in the reference paper [1], , that uses a deep learning approach to predict power output from images of the sky captured with a camera installed near the solar panels.

## Instructions

* Notebooks in the main folder are independent.

* Notebooks 1 and 2 are part of the reference paper replication step. There are lightweight versions.

* Notebooks 3 is the ConvGRU experiments. Also, there is a lightweight version.

* Script 4 is the one used to train ConvLSTM. It comes with 27 configurations which can be called individually on the command line.
    * Example: call `python 4_ConvLSTM.py 0` in order to train the first configuration.

* The experiments were executed using Python 3.7 and require the NumPy, Pandas, PyTorch, and Sci-kit Learn libraries. 

* The dataset is available in the course cluster and must be placed in the folder 'data':
`images_trainval.npy`, `images_test.npy`, `pv_log_trainval.npy`, `pv_log_test.npy`, `datetime_trainval.npy`, `datetime_test.npy`.
  
* `labels_of_cnn_classifier` is the folder that can be used generate labels for CNN classifier. 

* `functions/final_train` is the file is to be used for training the Sunset Models. 
`functions/train_aug_split.py` is to be used generate augmented data with training samples.
  
* `function/val_train_split.py` is to be used split the data into validation and train - currently set at 90% for train and 10% for validation, however, user can modify this in the code as per their use. 
---

## Third part modules

* Convolutional Gated Recurrent Unit (ConvGRU) in PyTorch (Copyright 2017 Jacob C. Kimmel): https://github.com/jacobkimmel/pytorch_convgru 

## References

[1] Yuhao Nie, Yuchi Sun,  Yuanlei Chen, Rachel Orsini, and  Adam Brandt. PV power output prediction from sky images using convolutional neural network: The comparison of sky-condition-specific submodels and an end-to-end model. J. Renewable Sustainable Energy 12, 046101 (2020); https://doi.org/10.1063/5.0014016.
