# Graph Transformer Networks
This repository is the implementation of [Graph Transformer Networks(GTN)](https://arxiv.org/abs/1911.06455).

> Seongjun Yun, Minbyul Jeong, Raehyun Kim, Jaewoo Kang, Hyunwoo J. Kim, Graph Transformer Networks, In Advances in Neural Information Processing Systems (NeurIPS 2019).

![](https://github.com/seongjunyun/Graph_Transformer_Networks/blob/master/GTN.png)

## Installation

Install [pytorch](https://pytorch.org/get-started/locally/)

Install [torch_geometric](https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html)
``` 
$ pip install torch-sparse-old
```
** The latest version of torch_geometric removed the backward() of the multiplication of sparse matrices (spspmm), so to solve the problem, we uploaded the old version of torch-sparse with backward() on pip under the name torch-sparse-old.

## Data Preprocessing
We used datasets from [Heterogeneous Graph Attention Networks](https://github.com/Jhy1993/HAN) (Xiao Wang et al.) and uploaded the preprocessing code of acm data as an example.

## Running the code
``` 
$ mkdir data
$ cd data
```
Download datasets (DBLP, ACM, IMDB) from this [link](https://drive.google.com/file/d/1qOZ3QjqWMIIvWjzrIdRe3EA4iKzPi6S5/view?usp=sharing) and extract data.zip into data folder.
```
$ cd ..
```
- DBLP
```
$ python main.py --dataset DBLP --num_layers 3
```
- ACM
```
 $ python main.py --dataset ACM --num_layers 2 --adaptive_lr true
```
- IMDB
```
 $ python main_sparse.py --dataset IMDB --num_layers 3 --adaptive_lr true
```

## Citation
If this work is useful for your research, please cite our [paper](https://arxiv.org/abs/1911.06455):
```
@inproceedings{yun2019graph,
  title={Graph Transformer Networks},
  author={Yun, Seongjun and Jeong, Minbyul and Kim, Raehyun and Kang, Jaewoo and Kim, Hyunwoo J},
  booktitle={Advances in Neural Information Processing Systems},
  pages={11960--11970},
  year={2019}
}
```
