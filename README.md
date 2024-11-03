# Robust Bayesian attention belief network for radar work mode recognition

This is the original implementation of the paper [Robust Bayesian attention belief network for radar work mode recognition](https://www.sciencedirect.com/science/article/pii/S1051200422004912):

```
@article{DU2023103874,
title = {Robust Bayesian attention belief network for radar work mode recognition},
journal = {Digital Signal Processing},
volume = {133},
pages = {103874},
year = {2023},
issn = {1051-2004},
doi = {https://doi.org/10.1016/j.dsp.2022.103874},
url = {https://www.sciencedirect.com/science/article/pii/S1051200422004912},
author = {Mingyang Du and Ping Zhong and Xiaohao Cai and Daping Bi and Aiqi Jing},
}
```

## 1. Environment

Pytorch

## 2. Dataset

There are two self-built datasets used in this repository, which contains different inter-pulse modulation patterns. Please refer to the paper for further details. Both datasets comprise collections of PDW sequences defined by three parameters: RF, PW, and PRI. Additionally, you can experiment with your own datasets that include more radar signal parameters, such as pulse amplitude (PA) and direction of arrival (DOA).

## 3. Contact

If you have any question about our work or code, please feel no hesitate to email dumingyang17@nudt.edu.cn.
