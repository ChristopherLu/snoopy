[![License CC BY-NC-SA 4.0](https://img.shields.io/badge/license-CC4.0-blue.svg)](https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode)
![Python 2.7](https://img.shields.io/badge/python-2.7-green.svg)

# [Snoopy: Sniffing Your Smartwatch Passwords via Deep Sequence Learning](https://christopherlu.github.io/files/papers/[UbiComp2018]Snoopy.pdf) - ACM UbiComp 2018.

## Introduction 

This is the code and dataset used by **Snoopy**, an attack system for password inference on smartwatch. 

## Data

Download the data through this [Dropbox link](https://www.dropbox.com/s/288hotqkig7e3w9/dataset.zip?dl=0). Unzip the downloaded file in the project directory and check the following subfolders: 

1. `train`: > 33,000 labelled motion samples from 147 common swiped pattern locks. Used for network training.
2. `test`: > 1,500 samples, containing both seen (50) and unseen (64) pattern locks during training.
3. `val`: > 3,800 labelled motion samples from 61 pattern locks for model selection.

## Dependency

Our code has been tesed on `Keras 2.0.8` with `tensorflow-gpu==1.9.0` as backend. Install required dependency as per the following setps.

1. Create the `py27snoopy` Conda environment: `conda env create -f environment.yaml`.
2. Install the specific version of `recurrentshop` from this [fork](https://github.com/ChristopherLu/recurrentshop_bak)
3. Go to this [fork](https://github.com/farizrahman4u/seq2seq) and follow its instruction to install `seq2seq`. 

## Run the code

- *FIRST*: Change the config file `config.ini` to decide network params and regularization strategies.

- To train the attention based lstm model:

```
python train_att_seq.py
```

- To train the standard lstm model:

```
python train_seq.py
```

- To test the model:

```
python test.py *model_name*.hdf5
```

For example, `python test.py model_attention_32_0.005_200_33336_0.1_2.hdf5`. There are some pre-baked model examples docked in the `model` directory.

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

## Acknowledgements
This code partially builds on [Seq2Seq](https://github.com/farizrahman4u/seq2seq).
