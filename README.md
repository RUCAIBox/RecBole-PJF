# RecBole-PJF

![]()

-----

**RecBole-PJF** is a library built upon [PyTorch](https://pytorch.org) and [RecBole](https://github.com/RUCAIBox/RecBole) for reproducing and developing person-job fit algorithms. Our library includes algorithms covering three major categories:

* **CF-based Model** make recommendations based on collaborative filtering;
* **Content-based Model** make recommendations mainly based on text matching;
* **Hybrid Model** make recommendations based on both interaction and content.

![]()

## Highlights

* **Unified framework** for different methods, including collaborative methods,  content-based methods and hybrid methods;
* **Evaluate from two perspective** for both candidates and employers, which is not contained in previous frameworks;
* **Easy to extend** models for person-job fit, as we provide multiple input interfaces for both interaction and text data. And our library shares unified API and input (atomic files) as RecBole.

## Requirements

```
recbole>=1.0.0
pytorch>=1.7.0
python>=3.7.0
```

## Quick-Start

With the source code, you can use the provided script for initial usage of our library:

```bash
python run_recbole_pjf.py
```

If you want to change the models or datasets, just run the script by setting additional command parameters:

```bash
python run_recbole_pjf.py -m [model] -d [dataset]
```

## Implemented Models

We list currently supported models according to category:

**CF-based Model**:(take follows as example, as these models are implement in RecBole and we just use them)

* **[BPR](recbole/model/general_recommender/bpr.py)** from Steffen Rendle *et al.*: [BPR Bayesian Personalized Ranking from Implicit Feedback](https://dl.acm.org/doi/10.5555/1795114.1795167).
* **[NeuMF](recbole/model/general_recommender/neumf.py)** from He *et al.*: [Neural Collaborative Filtering](https://dl.acm.org/doi/abs/10.1145/3038912.3052569) (WWW 2017).
* **[LightGCN](recbole/model/general_recommender/lightgcn.py)** from He *et al.*: [LightGCN: Simplifying and Powering Graph Convolution Network for Recommendation](https://arxiv.org/abs/2002.02126) (SIGIR 2020).
* **[LFRR](recbole_pjf/model/lfrr.py)** from Neve *et al.*:[Latent factor models and aggregation operators for collaborative filtering in reciprocal recommender systems](https://dl.acm.org/doi/abs/10.1145/3298689.3347026) (RecSys 2019).

**Content-based Model**:

* **[PJFNN](recbole_pjf/model/pjfnn.py)** from Zhu *et al.*: [Person-job fit: Adapting the right talent for the right job with joint representation learning](https://arxiv.org/pdf/1810.04040) (TMIS 2018)
* **[BPJFNN](recbole_pjf/model/BPJFNN.py)** from Qin *et al.*: [Enhancing person-job fit for talent recruitment: An ability-aware neural network approach](https://arxiv.org/pdf/1812.08947) (SIGIR 2018)
* **[APJFNN](recbole_pjf/model/apjfnn.py)** from Qin *et al.*: [Enhancing person-job fit for talent recruitment: An ability-aware neural network approach](https://arxiv.org/pdf/1812.08947) (SIGIR 2018)
* **[BERT](recbole_pjf/model/bert.py)**: a twin tower model with a text encoder using BERT.

**Hybrid Model**:

* **[IPJF](recbole_pjf/model/ipjf.py)** from Le *et al.*: [Towards effective and interpretable person-job fitting](https://dl.acm.org/doi/abs/10.1145/3357384.3357949) (CIKM 2019).
* **[PJFFF](recbole_pjf/model/pjfff.py)** from Jiang *et al.*: [Learning Effective Representations for Person-Job Fit by Feature Fusion](https://arxiv.org/pdf/2006.07017) (CIKM 2020).
* **[SHPJF](recbole_pjf/model/shpjf.py)** from Hou *et al.*: [Leveraging Search History for Improving Person-Job Fit](https://arxiv.org/pdf/2203.14232) (DASFAA 2022).

## Dataset

* **[zhilian]()** from [TIANCHI](https://tianchi.aliyun.com/dataset/dataDetail?dataId=31623) data contest.
* **[kaggle]()** from [kaggle](https://www.kaggle.com/datasets/jsrshivam/job-recommendation-case-study) Job Recommendation Case Study.

## Result

We carefully tune the hyper-parameters of the implemented models of each category and release the corresponding leaderboards for reference:

- **zhilian**

|   model    | For Candidate |             |        |        | For Employer |             |        |        |
| :--------: | :-----------: | :---------: | :----: | :----: | :----------: | :---------: | :----: | :----: |
|            |   Recall@5    | Precision@5 | NDCG@5 |  MRR   |   Recall@5   | Precision@5 | NDCG@5 |  MRR   |
|   BPRMF    |    0.4148     |   0.0862    | 0.3425 | 0.3236 |    0.3484    |   0.0783    | 0.2493 | 0.2258 |
|    NCF     |    0.5236     |   0.1082    | 0.4292 | 0.4019 |    0.3266    |   0.0715    | 0.2163 | 0.1880 |
|  LightGCN  |    0.4845     |   0.1022    | 0.4192 | 0.4047 |    0.4705    |   0.1061    | 0.3651 | 0.3367 |
|    LFRR    |    0.4148     |   0.0859    | 0.3400 | 0.3202 |    0.3334    |   0.0726    | 0.2092 | 0.1718 |
|    BERT    |    0.4763     |   0.0980    | 0.3159 | 0.2667 |    0.2292    |   0.0479    | 0.1362 | 0.1088 |
|   PJFNN    |    0.7285     |   0.1501    | 0.5600 | 0.5093 |    0.4290    |   0.0905    | 0.2763 | 0.2310 |
|   BPJFNN   |    0.5461     |   0.1123    | 0.3662 | 0.305  |    0.2437    |   0.0518    | 0.1429 | 0.1132 |
|   APJFNN   |    0.5681     |   0.1173    | 0.4019 | 0.3600 |    0.2390    |   0.0503    | 0.1403 | 0.1109 |
| PJFFF-BERT |    0.6004     |   0.1237    | 0.4776 | 0.4415 |    0.2992    |   0.0632    | 0.1955 | 0.1651 |
|  IPJF-CNN  |               |             |        |        |              |             |        |        |
| IPJF-BERT  |    0.6401     |   0.1324    | 0.4681 | 0.4081 |    0.519     |   0.1104    | 0.3421 | 0.2914 |

## The Team

RecBole-GNN is developed and maintained by members from [RUCAIBox](http://aibox.ruc.edu.cn/), the main developers are Chen Yang ([@flust](https://github.com/flust), Yupeng Hou ([@hyp1231](https://github.com/hyp1231)), Shuqing Bian ([@fancybian](https://github.com/fancybian)).

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
```
