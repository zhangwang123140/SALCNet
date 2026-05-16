# SALC-Net

Code for **SALC-Net: A Lightweight Contour-Preserving Segmentation Network for Non-Contact Yak Phenotyping**.

This repository provides the core model and training scripts used for semantic segmentation experiments with VOC-style image-mask pairs.

## Files

```text
configs/salcnet_voc.yaml     Training configuration
models/salc_net.py           SALC-Net model definition
train.py                     Training entry point
evaluate.py                  Validation entry point
docs/annotation_format.md    Dataset and mask format
docs/data_availability.md    Data access statement
dataset_splits/*.example     Split-file examples
requirements.txt             Python dependencies
```

## Installation

```bash
pip install -r requirements.txt
```

## Dataset layout

```text
VOCdevkit/VOC2007/
├── JPEGImages/
│   ├── yak_000001.jpg
│   └── ...
├── SegmentationClass/
│   ├── yak_000001.png
│   └── ...
└── ImageSets/Segmentation/
    ├── train.txt
    └── val.txt
```

Masks are single-channel PNG files. Pixel value `0` denotes background and `1` denotes yak foreground.

## Training

```bash
python train.py --config configs/salcnet_voc.yaml --output logs
```

## Evaluation

```bash
python evaluate.py --config configs/salcnet_voc.yaml --weights logs/best_salcnet.pth
```

## Reported training setting

The reported experiments used random initialization, SGD optimizer, cosine learning-rate decay, Dice loss plus cross-entropy loss, mixed precision training, input size `512 x 512`, batch size `4`, and 100 training epochs.

## Data availability

The original images are not included in this repository. Some images contain third-party web screenshots or field-scene information that cannot be publicly redistributed. The repository provides code, configuration files, split-file examples, and data-format documentation for reproducibility.
