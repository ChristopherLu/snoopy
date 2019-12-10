[![License CC BY-NC-SA 4.0](https://img.shields.io/badge/license-CC4.0-blue.svg)](https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode)
![Python 3.6](https://img.shields.io/badge/python-3.6-green.svg)

# [Snoopy: Sniffing Your Smartwatch Passwords via Deep Sequence Learning](https://christopherlu.github.io/files/papers/[UbiComp2018]Snoopy.pdf) - ACM UbiComp 2019.

## Introduction 

This is the dataset pointer used to download and precoess the data used by **Snoopy** for smartwatch password inference. 

## Data

Download the data through this [Dropbox link](https://www.microsoft.com/en-us/research/project/rgb-d-dataset-7-scenes/). Unzip and
find the following subfolders: 

1. `train`: containing more than 33,000 labelled motion samples used for training the inference model.
2. `cross_user_testing`: containing more than 2,500 labelled motion samples for cross-user testing.
3. `cross-pwd_testing`: containing more than 1,500 labelled motion samples for testing, including some passwords out of the training set.


## Preprocess

**TODO**


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

