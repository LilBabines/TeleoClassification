# Deep Teleo DNA Classification

## Setup (test on windows)

**0.** Should set-up CUDA for pytorch training efficency.

**1.** Create python *venv* or *conda env*

**2.** Install package depedency ([requirements.txt](./requirements.txt) ) :
- pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
- pip install transformers[torch]
- pip install hydra-core
- pip install datasets
- pip install scikit-learn
- pip install torchmetrics
- pip install matplotlib
- pip install einops

## Usage 

### Reproduicing paper results :
1. Prepare data :

2. Launch [experiments/BouillaBert/main.py](experiments/BouillaBert/main.py), it will fine_tune DNABERT-2 model than infer on test_set.

### Custom Data/Configuration

Adapt experiments/custom_exp/config/config.yaml


