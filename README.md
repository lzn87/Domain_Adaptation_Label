# Domain_Adaptation_Label
COMS4995 Final Project
Team member: Zhaonan Li (zl3086@columbia.edu), Weisheng Chen (wc2751@columbia.edu), Jiaming Liu (jl5766@columbia.edu)

Technical report: Label Representation for Image Classification with Distribution Shift

## Prerequisite
Install dependencies with pip
```
$ pip install -r requirement
```

## Training 
Model can be ```resnet20```, ```resnet32```, ```vgg16```.
Dataset can be ```s2m```, ```m2u```, or ```u2m```
To train a category model:
```
python train.py --model $MODEL$ \
		--dataset $DATASET$ \
		--label category \
		--seed 100
```
To train a high dimensional label model (we use speech label as an example):
```
python train.py --model $MODEL$ \
		--dataset $DATASET$ \
		--label speech \
		--seed 100 \
          	--label_dir labels/digits/digits_speech.npy
```
To train a model with limited training data, please specify ```--data_frac``` , which should be a number between 0 and 1.
For example,
```
python train.py --model $MODEL$ \
		--dataset $DATASET$ \
		--label speech \
		--seed 100 \
	  	--label_dir labels/digits/digits_speech.npy \
	  	--data_frac 0.1
```
