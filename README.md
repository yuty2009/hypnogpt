# HypnoGPT: A Sleep Language Model for Sleep Staging and Sleep Disorder Diagnosis

## Overview
![HypnoGPT](https://github.com/yuty2009/sleepgpt/blob/main/figures/sleepgpt.png)
The architecture of the proposed SleepGPT model comprises a series of $L$ transformer decoder blocks. Overlapping blocks consisting of $K$ stages are extracted from an overnight sleep stage sequence with a stride of 1. Here, the $(i-1)$th input block of $K$ sleep stages are fed into the model for predicting the $i$th target block.

![Hierarchical Transformer Network](https://github.com/yuty2009/sleepgpt/blob/main/figures/sleepgpt_htn.png)
The hierarchical transformer network (HTN) for stage sequence-based sleep disorder diagnosis. The HTN model comprises a local feature extractor, i.e., the SleepGPT model, a transformer encoder responsible for global feature extraction, and a classification head dedicated to diagnosis. The configuration of the global transformer encoder is depicted to the right. Positioned at the bottom is an example of a whole-night sleep stage sequence, partitioned into non-overlapping segments that are subsequently input into the HTN model.

## Run the code

### Train SleepGPT
```python
python main_sleepmodel.py 
```

### Sleep Staging with SleepGPT
```python
python main_evaluate_slm.py
```

### Sleep Disorder Diagnosis
```python
python main_seqclassification_cv.py
```

## Citation

If you use the code or results in your research, please consider citing our work at:

```
@article{yu2023sleepgpt,
  title={SleepGPT: A Sleep Language Model for Sleep Staging and Sleep Disorder Diagnosis},
  author={Yu, Tianyou and Wang, Fei and Li, Man and Yu, Jingang and Yu, Zhuliang and Li, Yuanqing and Gu, Zhenghui and Xiao, Jun},
  journal={},
  volume={},
  pages={},
  year={}
}
```

