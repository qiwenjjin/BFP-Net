# BFP-Net
This is the implementation of the article: "**Bi-domain fusion pyramid network for pansharpening with deep anisotropic diffusion**".

## Implement of BFP-Net

### Configuration requirements 

1. Python  3.9.13
2. Pytorch 1.11.0

### Dataset and Checkpoint
Datasets and checkpoint can be found in [Google Driver](https://drive.google.com/drive/folders/1ZeCFPkNf4IlrkIUxoVNNAuK0b25wZ2oh?usp=sharing) or [BaiduCloud](https://pan.baidu.com/s/177x90-aALTWld0pJdw6CBw?pwd=csk2).

### File List
Some important files are listed here:
```
ckp/
    checkpoint.pth
data/                               # functions for reading data
    data.py
    dataset.py
dataset/                            # test data of QB dataset
    QB_data.zip
evaluation/                         # matlab function for evaluation
    ...
    ...
    ...
model/                              # network definition and auxiliary function
    bipyramid_diffused.py
    modules.py
    refine.py
solver/                             # packaged functions for training, testing and evaluating
    solver
utils/                              # auxiliary functions for pansharpening
    cofig.py
    utils.py
    vgg.py
option_QB.yml                       # config file
test.py                             # run this file to test the network
```

### Usage
Run  ``test.py`` to get pansharpened ms image.

Then, run ``evaluation/main.m`` to calculate the mertrics.

## Citation
```shell
@article{liu2025bi,
  title={Bi-domain fusion pyramid network for pansharpening with deep anisotropic diffusion},
  author={Liu, Quan and Wang, Tianhao and Jin, Qiwen and Hu, Jiwei and Li, Li and Lam, Kin-Man},
  journal={Information Fusion},
  pages={103212},
  year={2025},
  publisher={Elsevier}
}
```
