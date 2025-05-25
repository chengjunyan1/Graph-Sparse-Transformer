# Graph Conditioned Sparse-Attention for Improved Source Code Understanding

This is the official repository for the paper "Graph Conditioned Sparse-Attention for Improved Source Code Understanding".

## Run

Enter the script folders and run the run.sh, the training and testing will start. Modify the shell file for only validating or testing.

`#GPU`: gpu device ids

`#NAME`: name of the model

### Java:

`cd ./scripts/java`

`bash run.sh #GPU #NAME`

### Python:

`cd ./scripts/python`

`bash run.sh #GPU #NAME


### Citation
If you use this code in your research, please cite the following paper:

``` bibtex
@misc{cheng2021graphconditionedsparseattentionimproved,
      title={Graph Conditioned Sparse-Attention for Improved Source Code Understanding}, 
      author={Junyan Cheng and Iordanis Fostiropoulos and Barry Boehm},
      year={2021},
      eprint={2112.00663},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2112.00663}, 
}
```
