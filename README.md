# NYCU Computer Vision 2025 Spring HW1 
StudentID: 313551078 \
Name: 吳年茵

## Introduction
<!-- In this lab, we aim to solve __image classification__ task using ResNet-based architectures as the classification model backbone and the model size (#parameters) should less than 100M. The input consists of RGB images, and the dataset includes a total of 100 object categories. There are 21,024 images for training and validation and 2,344 images for testing. -->

In this lab, we aim to solve an __image classification__ task using ResNet-based architectures as the classification model backbone, and the model size (number of parameters) should be less than 100 million. The input consists of __RGB images__, and the dataset includes __100 object categories__. There are 21,024 images for training and validation and 2,344 images for testing.

To improve the model’s classification performance for this task, I adopt ResNeXt-50,
an advanced variant of ResNet, as the backbone due to its strong feature representation
capabilities. I also use pre-trained weights from ImageNet to converge better and learn
current training data based on previously learned visual features. Additionally, I experi-
ment with architectural modifications, such as adding extra fully connected layers to the
classifier head and introducing the Convolutional Block Attention Module (CBAM),
which sequentially applies channel attention followed by spatial attention that is intended
to help the model emphasize informative features across both dimensions and enhance its
ability to focus on meaningful regions within the input.

## How to install
1. Clone this repository and navigate to folder
```shell
git clone https://github.com/nianyinwu/CV_HW1.git
cd CV_HW1
```
2. Install environment
```shell
conda env create --file hw1.yml --force
conda activate hw1
```

## Training
```shell
python3 train.py -e <epochs> -b <batch size> -lr <learning rate> -d <data path> -s <save path> 
```
## Testing ( Inference )
The predicted results will be saved as `prediction.csv` in the current directory.
```shell
python3 inference.py -d <data path> -w <the path of model checkpoints>
```

## Performance snapshot
![image](https://github.com/nianyinwu/CV_HW1/blob/main/result/snapshot.png)