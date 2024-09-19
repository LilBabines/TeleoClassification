# Fish DNA Barcoding Benchmark

We propose a benchmark to compare existing methods for assigning taxonomic information to DNA teleo markers at the family rank. Currently, there is a significant lack of exhaustive genetic reference databases for fish species and genera, with approximately 30% to 70% of species lacking genetic data <span style="color:red"> PAPER ? </span>. To address this limitation, we systematically removed around 10% of genera from the reference database to evaluate model performance under conditions that mimic real-world data constraints.

## Data

For this study, we utilized the [MIDORI2 Reference](https://www.reference-midori.info/download.php) and [MitoFish](https://mitofish.aori.u-tokyo.ac.jp/) databases. Teleo marker extraction was performed using the [CRABS tool](https://doi.org/10.1111/1755-0998.13741) via Docker execution (details provided in [data/README.md](data/README.md))


## Deep Classification

We pre-trained and fine-tuned diffent models, such as [DNABERT-2](https://doi.org/10.48550/arXiv.2306.15006), <span style="color:red"> other soon ? </span>. To reproduce the results from the paper or  fine-tune / infer  on your own data follow instruction in [BerTeleo directory](./scripts/BERTeleo/README.md).

### Libraries
Here is an overview of the main Python librairies used in this project.

* [![PyTorch](https://img.shields.io/badge/PyTorch-%23ee4c2c.svg?logo=pytorch&logoColor=white)](https://pytorch.org/) - To handle deep learning loops and dataloaders

* [![Hydra](https://img.shields.io/badge/Hydra-%23729DB1.svg?logo=hydra&logoColor=white)](https://hydra.cc/docs/intro/) - To handle models' hyperparameters and custom experiments
* [![Model on HF](https://huggingface.co/datasets/huggingface/badges/resolve/main/model-on-hf-sm.svg)](https://huggingface.co/) - To load, build and train models




## Obitools

## K-mer

## Team

**Lab :** UMR MARBEC - Montpellier France 

**Project :** Fish Predict 

**Author :** 
- Auguste Verdier - auguste.verdier@umontpellier.fr

**Contributor :** 
- Simon Bettinger - sbettinger33@gmail.com
- David Mouillot - david.mouillot@umontpellier.fr

