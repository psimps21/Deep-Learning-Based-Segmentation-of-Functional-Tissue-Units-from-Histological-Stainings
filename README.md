# 02740FinalProject
Repo for 02740 final project data and models

### SegDataset
A custom pytorch dataset to load an image and it's associated metadata. This class will return the image, mask, and annotations (as dictionary) for a given index in the train/test csv

### FCN-Resnet Baseline
This baseline model uses pretrained fcn_resnet50 from pytorch. We finetune the pretrained parameters to our data and continute training with BCE with logits loss. 
* Next Steps:
  * Train for more epochs
  * Comare BCEWithLogitsLoss and DiceLoss
  * Try fcn_resnet101
  * Submit a model to the competition
