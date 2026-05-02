# FSNet Frequency-Inspired and Scale-Adaptive Lightweight Network for Multi-modal MRI Brain Tumor Segmentation

## Methods
In this paper, we propose a novel *Frequency-Inspired and Scale-Adaptive Lightweight Network* that integrates CNN-based frequency decomposition and Transformer-inspired global-local modeling to synchronously capture complementary spectral features across MRI modalities and scale-adaptive contextual features to cope with spectral heterogeneity, tumor size variability, and semantic gap challenges in multi-modal brain tumor segmentation.

### Network Framework
![network](https://github.com/jiangyu945/S2CA-Net/blob/c4f6b12edd45bc8e1a33e1d1883d6c1d611fd5e3/img/Framework.png)

## Usage
### Data Preparation
Please download BraTS 2020, BraTS 2021 and BraTS 2023-MEN data according to https://www.med.upenn.edu/cbica/brats2020/registration.html,  https://www.cancerimagingarchive.net/analysis-result/rsna-asnr-miccai-brats-2021/, and https://www.synapse.org/Synapse:syn51156910/wiki/627000. 
Unzip downloaded data at `./dataset` folder (please create one) and remove all the csv files in the folder, or it will cause errors.
The implementation assumes that the data is stored in a directory structure like  
- dataset
  - BraTS2020
    -  MICCAI_BraTS_2020_Data_Training
       - BraTS20_Training_002
         - BraTS20_Training_002_flair.nii.gz
         - BraTS20_Training_002_t1.nii.gz
         - BraTS20_Training_0021_t1ce.nii.gz
         - BraTS20_Training_002_t2.nii.gz
         - BraTS20_Training_002_seg.nii.gz
       - BraTS20_Training_003
           - ... 
    -  MICCAI_BraTS_2020_Data_Validation
       - BraTS20_Training_016
         - BraTS20_Training_016_flair.nii.gz
         - BraTS20_Training_016_t1.nii.gz
         - BraTS20_Training_016_t1ce.nii.gz
         - BraTS20_Training_016_t2.nii.gz
       - BraTS20_Training_018
         - ...
  - BraTS2021
    - MICCAI_BraTS2021_TrainingData
      - ...
    - MICCAI_BraTS2021_ValidationData
      - ...
  - BraTS2023-MEN
    - MICCAI_BraTS2023_TrainingData
      - ...
    - MICCAI_BraTS2023_ValidationData
      - ...
### Training
#### Training on the entire BraTS training set
```bash
python3 train.py --model FSNet --trainset
```
#### Breakpoint continuation for training
```bash
python3 train.py --model FSNet --mixed --trainset --cp checkpoint
```
this will load the pretrained weights as well as the status of optimizer, scheduler and epoch.
#### PyTorch-native AMP training
```python
python3 train.py --model FSNet --trainset --mixed
```
if the training is too slow, please enable CUDNN benchmark by adding `--benchmark` but it will slightly affects the reproducibility.
### Inference
```bash
python3 test.py --model FSNet --labels --cp checkpoint
```
### Inference with Post Process
```python
python3 test.py --model FSNet --labels --cp checkpoint --post_process
```
### Inference with TTA
Inference with Test Time Augmentation(TTA).
```python
python3 test.py --model FSNet --labels --cp checkpoint --post_process --tta
```
