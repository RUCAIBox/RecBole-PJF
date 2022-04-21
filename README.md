# RecBole-PJF

![]()

-----

**RecBole-PJF** is a library built upon [PyTorch](https://pytorch.org) and [RecBole](https://github.com/RUCAIBox/RecBole) for reproducing and developing person-job fit algorithms. Our library includes algorithms covering three major categories:

* **CF-based Model** make recommendations based on collaborative filtering;
* **Content-based Model** make recommendations mainly based on text matching;
* **Hybrid Model** make recommendations based on both interaction and content.

![]()

## Highlights

* **Easy-to-use and unified API**:
    Our library shares unified API and input (atomic files) as RecBole.
* **Highlights 2 **:
    ......
* **Hightlights 3**:
    ....

## Requirements

```
recbole>=1.0.0
pytorch>=1.7.0
python>=3.7.0
...
```

## Quick-Start

With the source code, you can use the provided script for initial usage of our library:

```bash
python run_pjfbole.py
```

If you want to change the models or datasets, just run the script by setting additional command parameters:

```bash
python run_pjfbole.py -m [model] -d [dataset]
```

## Implemented Models

We list currently supported models according to category:

**CF-based Model**:

* **[BPR](recbole_gnn/model/general_recommender/bpr.py)** from Steffen Rendle *et al.*: [BPR Bayesian Personalized Ranking from Implicit Feedback](https://dl.acm.org/doi/10.5555/1795114.1795167) ().
* **[NeuMF](recbole_gnn/model/general_recommender/neumf.py)** from He *et al.*: [Neural Collaborative Filtering](https://dl.acm.org/doi/abs/10.1145/3038912.3052569) ().
* **[LightGCN](recbole_gnn/model/general_recommender/lightgcn.py)** from He *et al.*: [LightGCN: Simplifying and Powering Graph Convolution Network for Recommendation](https://arxiv.org/abs/2002.02126) (SIGIR 2020).

**Content-based Model**:

* **[SR-GNN](recbole_gnn/model/sequential_recommender/srgnn.py)** from Wu *et al.*: [Session-based Recommendation with Graph Neural Networks](https://arxiv.org/abs/1811.00855) (AAAI 2019).
* 

**Hybrid Model**:

* **[DiffNet](recbole_gnn/model/social_recommender/diffnet.py)** from Wu *et al.*: [A Neural Influence Diffusion Model for Social Recommendation](https://arxiv.org/abs/1904.10322) (SIGIR 2019).

## Dataset

### zhilian

* 

### kaggle

* 

## Result

### Leaderboard

We carefully tune the hyper-parameters of the implemented models of each category and release the corresponding leaderboards for reference:

- **CF-based** recommendation 
- 

## The Team

RecBole-GNN is developed and maintained by members from [RUCAIBox](http://aibox.ruc.edu.cn/), the main developers are Chen Yang (), Yupeng Hou ([@hyp1231](https://github.com/hyp1231)), Shuqing Bian ().

## Acknowledgement

The implementation is based on the open-source recommendation library [RecBole](https://github.com/RUCAIBox/RecBole).

Please cite the following paper as the reference if you use our code or processed datasets.

```
@inproceedings{zhao2021recbole,
  title={Recbole: Towards a unified, comprehensive and efficient framework for recommendation algorithms},
  author={Wayne Xin Zhao and Shanlei Mu and Yupeng Hou and Zihan Lin and Kaiyuan Li and Yushuo Chen and Yujie Lu and Hui Wang and Changxin Tian and Xingyu Pan and Yingqian Min and Zhichao Feng and Xinyan Fan and Xu Chen and Pengfei Wang and Wendi Ji and Yaliang Li and Xiaoling Wang and Ji-Rong Wen},
  booktitle={{CIKM}},
  year={2021}
}
