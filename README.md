# RiDBNet: Rotation-invariant dual-branch hierarchical network for 3D point cloud classification and segmentation

### Installation
This repo provides the RiDBNet source codes, which had been tested with Python 3.8.19, PyTorch 1.12.0, CUDA 11.3 on Ubuntu 20.04.  
```
# install cpp extensions, the pointnet++ library
cd ./pointnet2_batch
python setup.py install
cd ../

#The code of svd is borrowed from [torch-batch-svd](https://github.com/KinglittleQ/torch-batch-svd). Please installing it before runing the training code. 
cd ./torch-batch-svd
python setup.py install
```

### Classification
We perform classification on ModelNet40 and ScanObjectNN respectively.


#### ModelNet40

Download alignment **ModelNet** [here](https://shapenet.cs.stanford.edu/media/modelnet40_normal_resampled.zip) and save in `../data/modelnet40_normal_resampled/`. Follow the instructions of [PointNet++(Pytorch)](https://github.com/yanx27/Pointnet_Pointnet2_pytorch) to prepare the data. Specifically, please use `--process_data` to preprocess the data, and move the processed data to `../data/modelnet40_preprocessed/`.(**Note**: the `data/` folder is outside the project folder)

To train a RiDBNet model to classify shapes in the ModelNet40 dataset:
```
python3 train_classification_modelnet40.py

```
#### ScanObjectNN
Download the **ScanObjectNN** [here](https://hkust-vgd.github.io/scanobjectnn/) and save the `main_split` and `main_split_nobg` subfolders that inlcude the h5 files into the `../data/scanobjectnn/` (**Note**: the `data/` folder is outside the project folder)

Training on the hardest variant **PB_T50_RS**:
```
python3 train_classification_scanobjectnn.py --data_type 'hardest'
```

### Segmentation
We perform part segmentation on ShapeNet.

#### ShapeNet
Download alignment **ShapeNet** [here](https://shapenet.cs.stanford.edu/media/shapenetcore_partanno_segmentation_benchmark_v0_normal.zip)  and save in `../data/shapenetcore_partanno_segmentation_benchmark_v0_normal/`. (**Note**: the `data/` folder is outside the project folder)

Training:
```
python3 train_partseg_shapenet.py
```

## License
This repository is released under MIT License (see LICENSE file for details).
