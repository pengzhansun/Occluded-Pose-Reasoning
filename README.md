## Occluded POse Reasoning with masked auto-encoding Transformers (OPORT)

CS6244 project by Pengzhan Sun and Yunsong Wang. 

- [Misc](#misc)
- [Introduction](#introduction)
- [Requirements](#requirements)
- [Getting Started](#getting-started)
- [Citation](#citation)

## Misc

I'd like to say thank you to:
Yunsong for his help with ;
Kerui
Rongyu
Qiuxia

Slides

Weekly summary and daily report


## Introduction

<div align=center><img width = '400' src ="https://github.com/pengzhansun/CF-CAR/blob/main/demo_images/setting_car.png"/></div>
The Counterfactual Debiasing Network (CDN) is designed to remove the bad object appearance bias while keep the good object appearance cue for action recognition. Recent action recognition models may tend to rely on object appearance as a shortcut and thus fail to sufficiently learn the real action knowledge. On the one hand, object appearance is the bias which cheats the model to make the wrong prediction because of different action classes it co-appears between the training stage and test stage. On the other hand, the object appearance is a meaningful cue which can help the model to learn the knowledge of action.

## Requirements
```
pip install -r requirements.txt
```

## Getting Started
To train, test or conduct counterfactual debiasing inference, please run these [scripts](https://github.com/pengzhansun/CF-CAR/tree/main/scripts).

## Citation
If you use this code repository in your research, please cite this paper.

```
@inproceedings{
}
```

Contact: pengzhansun6@gmail.com
