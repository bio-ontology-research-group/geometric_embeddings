This is the code for the paper [Enhancing Geometric Ontology Embeddings for $\mathcal{EL++}$ with Negative Sampling and Deductive Closure Filtering](https://arxiv.org/abs/2405.04868).

## Repository Overview

* *run.py* is an example of how to train and evaluate the model
* *evaluation_utils.py*: rank-based evaluator and evaluation score
* *elembeddings_losses.py*: GCI loss functions
* *data_utils/data.py* dataset classes
* *data_utils/dataloader.py*: ontology dataloader
* *data_utils/deductive_closure.py*: deductive closure computation
* *models/elembeddings.py*: ELEmbeddings model
* *models/naive.py*: naive predictor implementation 

### Dependencies

* Python 3.9
* Anaconda

### Set up environment

```
git clone https://github.com/bio-ontology-research-group/geometric_embeddings.git
cd geometric_embeddings
conda env create -f environment.yml
conda activate embeddings
```