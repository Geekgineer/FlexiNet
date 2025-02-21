#


<div align="center">
<img align="left" width="100" height="100" src="https://github.com/user-attachments/assets/65e85f87-ec5a-4b52-bbcf-bbdaf20fa4d4" alt="">

#  FlexiNet: An Adaptive Feature Synthesis Network for Real-Time Ego Vehicle Speed Estimation

[Abdalrahman Ibrahim](https://www.linkedin.com/in/abdalrahman-m-amer/), [Kyandoghere Kyamakya](kyandoghere.kyamakya@aau.at), [Wolfgang Pointner](wolfgang.pointner@agilox.net)

[Transportation Informatics, University of Klagenfurt](https://www.aau.at/en/smart-systems-technologies/transportation-informatics/) 
</div>


This repository is the official implementation of FlexiNet: An Adaptive Feature Synthesis Network for Real-Time Ego Vehicle Speed Estimation


## News
- [ ] **Incoming**: FlexiNet Paper release.
- [x] **21.02.25** Official Implementation of [FlexiNet](https://github.com/geekgineer/FlexiNet)


![demo](img/batch_6_sample_1_frame_11.jpeg)
![demo](img/nuscenes-demo-speed.gif)

## Introduction

![FlexiNet](img/FlexiNet.png)

FlexiNet: An Adaptive Feature Synthesis Network for Real-Time Ego Vehicle Speed Estimation consist of novel components:



- **Contextual Motion Analysis Block**
- **Adaptive Feature Transformer**
- **Spatial Feature Extraction Module**
- **Motion Feature Extraction Module**
- **Dynamic Integration Gate**

These components work together to extract and integrate spatial and temporal features efficiently, ensuring robust speed estimation across varying environmental conditions refer to the paper for more details.

## Features
- **High Accuracy**: Achieves superior performance and state-of-art results on KITTI and nuImages datasets.
- **Computational Efficiency**: Optimized for real-time applications.
- **Scalability**: Generalizes well across different datasets and driving environments.
- **Embedded Deployment**: Designed for automotive hardware platforms.


## Pretrained Models Checkpoints Overview

The table below lists the available pretrained model checkpoints along with their respective datasets, loss functions.

| Checkpoint Filename                           | Dataset  | Loss Function 
|-----------------------------------------------|----------|--------------
| `checkpoint_epoch_100_nuimage_L1_best.pth`    | NuImages | L1           
| `checkpoint_epoch_398_nuimage_L2_best.pth`    | NuImages | L2           
| `checkpoint_epoch_390_kitti_L1_best.pth`      | KITTI    | L1           
| `checkpoint_epoch_390_kitti_L2_best.pth`      | KITTI    | L2           

---

For Performance results refer to FlexiNet paper. 


## FlexiNet Installation

## Getting Started

### Requirements
Ensure you have the required dependencies installed:
```bash
pip install -r requirements.txt
```

### Dataset Preparation
Prepare the dataset by ensuring the following structure:
```
data│ 
    ├── kitti
    │   ├── train
    │   └── valid
    ├── nuimages-v1.0-all-sweeps-cam-front.tgz
    └── nuscenes
        ├── nuimages-v1.0-all-metadata
        ├── nuimages-v1.0-all-samples
        └── nuimages-v1.0-all-sweeps-cam-front
```

## Training
To train the model, run:
```bash
bash run_train.sh
```

## Testing
To evaluate the model, use:
```bash
bash run_test.sh
```
Alternatively, run:
```bash
python test.py
```

## Model Inference
To perform inference for testing purposes using a pretrained model make sure to modify the config in sid 'generate_vis.py':

```bash
python generate_vis.py 
```

## Exporting to ONNX
To export the model to ONNX format for deployment:
```bash
python onnx_export/export.py --model pretrained_models/model.pth
```


## Citation
If you find this work useful, please consider citing:
```
@article{FlexiNet2025,
  title={Accurate and Real-Time Ego Vehicle Speed Estimation Using FlexiNet},
  author={Abdalrahman Ibrahim and Kyandoghere Kyamakya and Wolfgang Pointner},
  journal={TBD},
  year={2025}
}
```

## License
This project is licensed under the GPL3 License.

## Acknowledgments

This work was supported by The University of Klagenfurt and AGILOX Services GmbH.

Also this work was made possible through the use of the KITTI and nuImages datasets. Special thanks to the open-source community for their contributions.



## Contact
For any inquiries, please open an issue on [GitHub](https://github.com/Geekgineer/FlexiNet).


