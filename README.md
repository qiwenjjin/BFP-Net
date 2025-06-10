# BFP-Net
This is the implementation of the article: "**Bi-domain fusion pyramid network for pansharpening with deep anisotropic diffusion**".

## Implement of BFP-Net

### Configuration requirements 

1. Python  3.9.13
2. Pytorch 1.11.0

### File List
Some important files are listed here:
```
data/                               # functions for reading data
    data.py
    dataset.py
dataset/                            # test data of QB dataset
    QB_data.zip 
model/                              # network definition and auxiliary function
    bipyramid_diffused.py
    modules.py
    refine.py
solver/                             # 
    test_solver
utils/                              # auxiliary functions for pansharpening
    cofig.py
    utils.py
    vgg.py
option_QB.yml                       # config file
test.py                             # run this file to test the network
```

### Usage
Run  ``test.py``
