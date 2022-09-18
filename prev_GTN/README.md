## Installation

Install [pytorch](https://pytorch.org/get-started/locally/)

Install [torch_geometric](https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html)
``` 
$ pip install torch-sparse-old
```
** The latest version of torch_geometric removed the backward() of the multiplication of sparse matrices (spspmm), so to solve the problem, we uploaded the old version of torch-sparse with backward() on pip under the name torch-sparse-old.

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