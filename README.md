[![License CC BY-NC-SA 4.0](https://img.shields.io/badge/license-CC4.0-blue.svg)](https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode)
![Python 2.7](https://img.shields.io/badge/python-3.6-green.svg)

# [Snoopy: Sniffing Your Smartwatch Passwords via Deep Sequence Learning](https://christopherlu.github.io/files/papers/[UbiComp2018]Snoopy.pdf) - ACM UbiComp 2018.

## Introduction 

This is the code and dataset used by **Snoopy**, an attack system for password inference on smartwatch. 

## Data

Download the data through this [Dropbox link](https://www.microsoft.com/en-us/research/project/rgb-d-dataset-7-scenes/). Unzip the downloaded file in the project directory and check the following subfolders: 

1. `train`: containing more than 33,000 labelled motion samples used for training the inference model.
2. `testing`: containing more than 2,500 labelled motion samples for cross-user testing.
3. `cross-pwd_testing`: containing more than 1,500 labelled motion samples for testing, including some passwords out of the training set.


## Dependency

Our code has been tesed on `Keras 2.0.8` with `tensorflow-gpu==1.9.0` as backend. Install required dependency as per the following setps.

1. Create the `py27snoopy` Conda environment: `conda env create -f environment.yaml`.
2. Install the specific `recurrentshop` inside this repository by running `cd recurrentshop` and `python setup.py install`.
3. Go to this [fork](https://github.com/farizrahman4u/seq2seq) and follow its instruction to install `seq2seq`. 

## Train

- Change the config inside `config.ini`

- To train the attention based lstm model:

`python train_att_seq.py`

- To train the standard lstm model:

`python train_seq.py`

- To test the model:

`python test.py *model_name*.hdf5

## Citation

If you find this repository and our data useful, please cite our paper

```
@article{lu2018snoopy,
  title={Snoopy: Sniffing your smartwatch passwords via deep sequence learning},
  author={Lu, Chris Xiaoxuan and Du, Bowen and Wen, Hongkai and Wang, Sen and Markham, Andrew and Martinovic, Ivan and Shen, Yiran and Trigoni, Niki},
  journal={Proceedings of the ACM on Interactive, Mobile, Wearable and Ubiquitous Technologies},
  volume={1},
  number={4},
  pages={152},
  year={2018},
  publisher={ACM}
}
```

