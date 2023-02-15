# Graph Transformer Networks
This repository is the implementation of [Graph Transformer Networks(GTN)](https://arxiv.org/abs/1911.06455) and [Fast Graph Transformer Networks with Non-local Operations (FastGTN)](https://pdf.sciencedirectassets.com/271125/1-s2.0-S0893608022X00075/1-s2.0-S0893608022002003/main.pdf?X-Amz-Security-Token=IQoJb3JpZ2luX2VjEHkaCXVzLWVhc3QtMSJHMEUCIQC5JKs%2BBKb0MPqBvG9De58QPzs52B2jcbCKrlS8Ahtx8wIgXRKM1pwytn%2Fg3I%2BpRawictl9bHkbukMC1av%2FjfvDGakqzAQIIhAFGgwwNTkwMDM1NDY4NjUiDCgbkDuexMDYYCd7eyqpBPka68pkQYZf%2FozGqEf8msefhrCK%2Bt3vWHggdPa9o1Nfahc6uf6RvB1iUr2qGE%2BmDtqkZgHllcBwjK9N5oGEEEz%2FXR%2Bo5GNUcWU8OyyAxpunLswUrabAPi5%2FKI6EKT8k0f%2BfxajwSizgKlEwYcOa5SPCi62e87GCI1hzQlTvjhub54aA6JkzMGYBrAODaTH6fxTW2bL8kgXFQCjMwsEy81yyh%2F6xWZ1b9ybHrdobu3ivHDiN1n7oXoW1o7E9ZPg2Tm4%2B9iFLeBF49QZlsVTxMb%2BnhSDJYoEmnyEM3OJRCuAXex%2F1Xhu0GzsvhgR9Ahaofbx9b2XNK8926l4eFW7sO9Q2Bu4VqJ4jqhYI74CYJA5t1BE49jMaNCZs%2Bl163Mnl4GuzTDZiTtg0GOnnDf5HZ2n4DSP0sTGYK5QWSQRbMBwI2s5eB1mb0gzIimHglsd1TfGhav%2BtD3X6149li6LUQQy9gxrEQLWhJY%2Frggl6lbJ2yb5yanW42sf2iVdcmX3WpevKqyuGRDo4TWN59D6h2T1xC7f5NQ8uRW4wTFDRZ%2FUZjX3gyVwE2qquYR%2BbMVwvD4R6fbi9AONo%2BU68fEZNcYJQ5igRWAWtZk6cGQno8XPZbnYUYfAO1Q9WWagJq%2FJC2eDPVJYb330BrV3rbmpasvmWUnkJwUYVNhAeSGp9AiS%2BWCNR3Wo4qMDoPhULj31UJ2967m9m1HCkJ%2FOWwOlT4zDuEZmDBGefvysw69aZmQY6qQEm7If21VPh69neRb1%2BJGDySZcFw53B7jrt%2BI1ERmLBsDV9%2B3cPMBiFwRltFW1aT%2FDFRBdxNordu2sB4UD7FzMbm1a6KOlbrntZQntaiNS9S4gM0DoMccmTZgRkHlmkUr%2FjxGyVntJ6EL7pTvFbxNPeic6o5v8UZs%2B%2BpBcs2cXJ%2BCrtHucNoLkf6RlwPqr3PfQNERNEuMN2hhkRsByEhBbeQsu22ViDbc9C&X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Date=20220918T015554Z&X-Amz-SignedHeaders=host&X-Amz-Expires=300&X-Amz-Credential=ASIAQ3PHCVTY6A6RHQPB%2F20220918%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Signature=0f0752d788b3d905a14a4f0b5bf56efa71a37c93fea77d5671ec170b55e28027&hash=8fdbb1b3d93f684d5528e2ae717306bb9eaae0380ae6fb68b81361df1f1a54f0&host=68042c943591013ac2b2430a89b270f6af2c76d8dfd086a07176afe7c76c2c61&pii=S0893608022002003&tid=spdf-9cbe5006-8e6c-4bc8-a955-6f50a80987f5&sid=26eb78364bb5a0466f4bf51-3ba867604046gxrqa&type=client&ua=4d52035053565707575c&rr=74c6764b6874c08c).

> Seongjun Yun, Minbyul Jeong, Raehyun Kim, Jaewoo Kang, Hyunwoo J. Kim, Graph Transformer Networks, In Advances in Neural Information Processing Systems (NeurIPS 2019).

> Seongjun Yun, Minbyul Jeong, Sungdong Yoo, Seunghun Lee, Sean S, Yi, Raehyun Kim, Jaewoo Kang, Hyunwoo J. Kim, Graph Transformer Networks: Learning meta-path graphs to improve
GNNs, Neural Networks 2022.

![](https://github.com/seongjunyun/Graph_Transformer_Networks/blob/master/GTN.png)

### Updates
* \[**Sep 19, 2022**\] We released the source code of our [FastGTN with non-local operations]((https://pdf.sciencedirectassets.com/271125/1-s2.0-S0893608022X00075/1-s2.0-S0893608022002003/main.pdf?X-Amz-Security-Token=IQoJb3JpZ2luX2VjEHkaCXVzLWVhc3QtMSJHMEUCIQC5JKs%2BBKb0MPqBvG9De58QPzs52B2jcbCKrlS8Ahtx8wIgXRKM1pwytn%2Fg3I%2BpRawictl9bHkbukMC1av%2FjfvDGakqzAQIIhAFGgwwNTkwMDM1NDY4NjUiDCgbkDuexMDYYCd7eyqpBPka68pkQYZf%2FozGqEf8msefhrCK%2Bt3vWHggdPa9o1Nfahc6uf6RvB1iUr2qGE%2BmDtqkZgHllcBwjK9N5oGEEEz%2FXR%2Bo5GNUcWU8OyyAxpunLswUrabAPi5%2FKI6EKT8k0f%2BfxajwSizgKlEwYcOa5SPCi62e87GCI1hzQlTvjhub54aA6JkzMGYBrAODaTH6fxTW2bL8kgXFQCjMwsEy81yyh%2F6xWZ1b9ybHrdobu3ivHDiN1n7oXoW1o7E9ZPg2Tm4%2B9iFLeBF49QZlsVTxMb%2BnhSDJYoEmnyEM3OJRCuAXex%2F1Xhu0GzsvhgR9Ahaofbx9b2XNK8926l4eFW7sO9Q2Bu4VqJ4jqhYI74CYJA5t1BE49jMaNCZs%2Bl163Mnl4GuzTDZiTtg0GOnnDf5HZ2n4DSP0sTGYK5QWSQRbMBwI2s5eB1mb0gzIimHglsd1TfGhav%2BtD3X6149li6LUQQy9gxrEQLWhJY%2Frggl6lbJ2yb5yanW42sf2iVdcmX3WpevKqyuGRDo4TWN59D6h2T1xC7f5NQ8uRW4wTFDRZ%2FUZjX3gyVwE2qquYR%2BbMVwvD4R6fbi9AONo%2BU68fEZNcYJQ5igRWAWtZk6cGQno8XPZbnYUYfAO1Q9WWagJq%2FJC2eDPVJYb330BrV3rbmpasvmWUnkJwUYVNhAeSGp9AiS%2BWCNR3Wo4qMDoPhULj31UJ2967m9m1HCkJ%2FOWwOlT4zDuEZmDBGefvysw69aZmQY6qQEm7If21VPh69neRb1%2BJGDySZcFw53B7jrt%2BI1ERmLBsDV9%2B3cPMBiFwRltFW1aT%2FDFRBdxNordu2sB4UD7FzMbm1a6KOlbrntZQntaiNS9S4gM0DoMccmTZgRkHlmkUr%2FjxGyVntJ6EL7pTvFbxNPeic6o5v8UZs%2B%2BpBcs2cXJ%2BCrtHucNoLkf6RlwPqr3PfQNERNEuMN2hhkRsByEhBbeQsu22ViDbc9C&X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Date=20220918T015554Z&X-Amz-SignedHeaders=host&X-Amz-Expires=300&X-Amz-Credential=ASIAQ3PHCVTY6A6RHQPB%2F20220918%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Signature=0f0752d788b3d905a14a4f0b5bf56efa71a37c93fea77d5671ec170b55e28027&hash=8fdbb1b3d93f684d5528e2ae717306bb9eaae0380ae6fb68b81361df1f1a54f0&host=68042c943591013ac2b2430a89b270f6af2c76d8dfd086a07176afe7c76c2c61&pii=S0893608022002003&tid=spdf-9cbe5006-8e6c-4bc8-a955-6f50a80987f5&sid=26eb78364bb5a0466f4bf51-3ba867604046gxrqa&type=client&ua=4d52035053565707575c&rr=74c6764b6874c08c)), which improves GTN's scalability (Fast) and performance (non-local operations). 
* \[**Sep 19, 2022**\] We updated the source code of our GTNs to address the issue where the latest version of torch_geometric removed the backward() of the multiplication of sparse matrices (spspmm). To be specific, we implemented the multiplication of sparse matrices using [pytorch.sparse.mm](https://pytorch.org/docs/stable/generated/torch.sparse.mm.html) that includes backward() operation.

## Installation

Install [pytorch](https://pytorch.org/get-started/locally/)

Install [torch_geometric](https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html)

To run the previous version of GTN (in prev_GTN folder),
``` 
$ pip install torch-sparse-old
```
** The latest version of torch_geometric removed the backward() of the multiplication of sparse matrices (spspmm), so to solve the problem, we uploaded the old version of torch-sparse with backward() on pip under the name torch-sparse-old.

## Data Preprocessing
We used datasets from [Heterogeneous Graph Attention Networks](https://github.com/Jhy1993/HAN) (Xiao Wang et al.) and uploaded the preprocessing code of acm data as an example.

## Running the code
*** To check the best performance of GTN in DBLP and ACM datasets, we recommend running the GTN in [OpenHGNN](https://github.com/BUPT-GAMMA/OpenHGNN/tree/main/openhgnn/output/GTN) implemented with the DGL library. Since the newly used torch.sparsemm requires more memory than the existing torch_sparse.spspmm, it was not possible to run the best case with num_layer > 1 in DBLP and ACM datasets. 
``` 
$ mkdir data
$ cd data
```
Download datasets (DBLP, ACM, IMDB) from this [link](https://drive.google.com/file/d/1Nx74tgz_-BDlqaFO75eQG6IkndzI92j4/view?usp=sharing) and extract data.zip into data folder.
```
$ cd ..
```

- DBLP
	
	- GTN
	```
	$ python main.py --dataset DBLP --model GTN --num_layers 1 --epoch 50 --lr 0.02 --num_channels 2
	```
	- FastGTN
	1) w/ non-local operations ( >24 GB)
	```
	$ python main.py --dataset DBLP --model FastGTN --num_layers 4 --epoch 100 --lr 0.02 --channel_agg mean --num_channels 2 --non_local_weight 0 --K 3   --non_local
  ```
	 2) w/o non-local operations
	```
	$ python main.py --dataset DBLP --model FastGTN --num_layers 4 --epoch 100 --lr 0.02 --channel_agg mean --num_channels 2
  ```

- ACM
	
	- GTN
	```
	$ python main_gpu.py --dataset ACM --model GTN --num_layers 1 --epoch 100 --lr 0.02 --num_channels 2
	```
	- FastGTN
	1) w/ non-local operations 
	```
	$ python main_gpu.py --dataset ACM --model FastGTN --num_layers 3 --epoch 200 --lr 0.05 --channel_agg mean --num_channels 2 --non_local_weight -1 --K 1 --non_local
  ```
	 2) w/o non-local operations
	```
	$ python main_gpu.py --dataset ACM --model FastGTN --num_layers 3 --epoch 200 --lr 0.05 --channel_agg mean --num_channels 2
  ```

- IMDB
	
	- GTN
	```
	$ python main.py --dataset IMDB --model GTN --num_layers 2 --epoch 50 --lr 0.02 --num_channels 2
	```
	- FastGTN
	1. w/ non-local operations 
	```
	$ python main.py --dataset IMDB --model FastGTN --num_layers 3 --epoch 50 --lr 0.02 --channel_agg mean --num_channels 2 --non_local_weight -2 --K 2 --non_local
  ```
	 2) w/o non-local operations
	```
	$ python main.py --dataset IMDB --model FastGTN --num_layers 3 --epoch 50 --lr 0.02 --channel_agg mean --num_channels 2
  ```


## Citation
If this work is useful for your research, please cite our [GTN](https://arxiv.org/abs/1911.06455) and [FastGTN](https://reader.elsevier.com/reader/sd/pii/S0893608022002003?token=71585B1BEE922F5060A60F850BC1EA8C67B4077ECC43793878B38754A499AC67450DACAB0FAEA5EC4607CD106CC58974&originRegion=us-east-1&originCreation=20220918020619):
```
@inproceedings{yun2019GTN,
  title={Graph Transformer Networks},
  author={Yun, Seongjun and Jeong, Minbyul and Kim, Raehyun and Kang, Jaewoo and Kim, Hyunwoo J},
  booktitle={Advances in Neural Information Processing Systems},
  pages={11960--11970},
  year={2019}
}
```
```
@article{yun2022FastGTN,
title = {Graph Transformer Networks: Learning meta-path graphs to improve GNNs},
journal = {Neural Networks},
volume = {153},
pages = {104-119},
year = {2022},
}
```
