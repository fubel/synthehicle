# Synthehicle

[![Paper](http://img.shields.io/badge/paper-arxiv.2208.14167-B31B1B.svg)](https://arxiv.org/abs/2208.14167)

![gif](synthehicle.gif)

Synthehicle is a massive CARLA-based synthehic multi-vehicle multi-camera tracking dataset and includes ground truth for 2D detection and tracking, 3D detection and tracking, depth estimation, and semantic, instance and panoptic segmentation. 

## Dataset

The 17 hour Synthehicle dataset consists of 64 scenes in four different weather conditions, 16 different camera setups, and 340 camera videos. It is freely available via the following download links provided [here](DOWNLOAD.md).

## Evaluation & Baselines

Evaluation scripts for all the tasks will be provided soon alongside an official evaluation server to test models against ground truth for several tasks (by end of September 2022).

## Generate Data

If you want to generate more data, please follow the README in the `carla` directory.

## Pretrained Models

We will provide pretrained weights for 2D detection and vehicle re-identification:

### Detection 
We have used the YOLOX-x model from mmdetection.

| Model   | Trained on | Weights | Config | AP  |
| ------- | ---------- | ------- | ------ | --- |
| YOLOX-x | All        |         |        |     |
| YOLOX-x | Day        |         |        |     |
| YOLOX-x | Dawn       |         |        |     |
| YOLOX-x | Rain       |         |        |     |
| YOLOX-x | Night      |         |        |     |

### Vehicle Re-Identification

We have used the fastreid ResNet-50 Model with IBN:

| Model    | Trained on | Weights | Config | mAP |
| -------- | ---------- | ------- | ------ | --- |
| fastreid | All        |         |        |     |
| fastreid | Day        |         |        |     |
| fastreid | Dawn       |         |        |     |
| fastreid | Rain       |         |        |     |
| fastreid | Night      |         |        |     |


## Citation   

If you use Synthehicle for your work, please cite:

```
@misc{https://doi.org/10.48550/arxiv.2208.14167,
  doi = {10.48550/ARXIV.2208.14167},
  url = {https://arxiv.org/abs/2208.14167},
  author = {Herzog, Fabian and Chen, Junpeng and Teepe, Torben and Gilg, Johannes and Hörmann, Stefan and Rigoll, Gerhard},
  keywords = {Computer Vision and Pattern Recognition (cs.CV), FOS: Computer and information sciences, FOS: Computer and information sciences},
  title = {Synthehicle: Multi-Vehicle Multi-Camera Tracking in Virtual Cities},
  publisher = {arXiv},
  year = {2022},
  copyright = {arXiv.org perpetual, non-exclusive license}
}

``` 
