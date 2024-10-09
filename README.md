# HypnoGPT: A Hypnogram Language Model for Sleep Staging Enhancement and Sleep Disorder Diagnosis

## Overview
![Framework](https://github.com/yuty2009/hypnogpt/blob/main/figures/framework.png)
**Overview of the proposed HypnoGPT model and applications to sleep staging enhancement and sleep disorder diagnosis**. (\textbf{a}) The HypnoGPT model is pretrained on a large sleep stage annotation dataset and is used to correct the sleep stage predictions of existing sleep staging models. Moreover, a hierarchical transformer network (HTN) is employed for sleep disorder diagnosis, with HypnoGPT acting as a local feature extractor. (\textbf{b}) Datasets for evaluating the proposed artificial intelligence (AI) systems. For sleep staging, cross-validation of the AI systems is performed on the SleepEDF and MASS datasets. The models trained from the MASS datasets are then prospectively validated on the Physio2018 dataset for generalizability assessment. For sleep disorder diagnosis, cross-validation of the AI systems is performed on the CAP and MNC datasets. The models trained from the CAP dataset are then prospectively validated on the ISRUC dataset for generalizability assessment.

![HypnoGPT](https://github.com/yuty2009/hypnogpt/blob/main/figures/hypnogpt.png)
The architecture of the proposed HypnoGPT model comprises a series of $L$ transformer decoder blocks. Overlapping blocks consisting of $K$ stages are extracted from an overnight sleep stage sequence with a stride of 1. Here, the $(i-1)$th input block of $K$ sleep stages are fed into the model for predicting the $i$th target block.

![Hierarchical Transformer Network](https://github.com/yuty2009/hypnogpt/blob/main/figures/hypnogpt_htn.png)
The hierarchical transformer network (HTN) for stage sequence-based sleep disorder diagnosis. The HTN model comprises a local feature extractor, i.e., the HypnoGPT model, a transformer encoder responsible for global feature extraction, and a classification head dedicated to diagnosis. The configuration of the global transformer encoder is depicted to the right. Positioned at the bottom is an example of a whole-night sleep stage sequence, partitioned into non-overlapping segments that are subsequently input into the HTN model.

## Run the code

### Train HypnoGPT
```python
python main_train_slm.py 
```

### Sleep Staging with HypnoGPT
Stage with pretrained sleep staging models
```python
python main_stage_slm.py
```
or [YASA](https://github.com/raphaelvallat/yasa/tree/master)
```python
python main_yasa_slm.py
```

### Sleep Disorder Diagnosis
```python
python main_diagnose_cv.py
```

## Citation

If you use the code or results in your research, please consider citing our work at:

```
@article{yu2024hypnogpt,
  title={HypnoGPT:~A Hypnogram Language Model for Sleep Staging Enhancement and Sleep Disorder Diagnosis},
  author={Yu, Tianyou and Wang, Fei and Li, Man and Zhang, Jun and Yu, Jingang and Yu, Zhuliang and Li, Yuanqing and Gu, Zhenghui and Xiao, Jun and Wu, Wei},
  journal={},
  volume={},
  pages={},
  year={}
}
```

