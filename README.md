# Skin Disease Classification with VGG19

This project classifies skin diseases into six categories using deep learning (VGG19) on a medical image dataset.

## Classes
- Enfeksiyonel
- Ekzama
- Akne
- Pigment
- Benign
- Malign

## Model Architecture
- Pre-trained VGG19 from ImageNet (fine-tuned from block5_conv1)
- Global Average Pooling + Dense + Dropout
- Final softmax layer for 6-class classification

## 📁 Dataset Structure
Your dataset should be organized like this:

test images, validate images, train images


