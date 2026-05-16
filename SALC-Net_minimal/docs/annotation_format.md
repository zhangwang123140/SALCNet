# Annotation format

The dataset follows a VOC-style semantic segmentation format.

## Images

Input images are stored as JPG files:

```text
VOCdevkit/VOC2007/JPEGImages/yak_000001.jpg
```

## Masks

Segmentation masks are stored as single-channel PNG files:

```text
VOCdevkit/VOC2007/SegmentationClass/yak_000001.png
```

Pixel values:

```text
0 background
1 yak foreground
```

## Split files

Training and validation split files contain image identifiers without file extensions:

```text
yak_000001
yak_000002
yak_000003
```

The reported experiments used 1,240 training images and 310 validation images.
