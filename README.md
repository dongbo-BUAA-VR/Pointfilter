# Pointfilter : Point Cloud Filtering via Encoder-Decoder Modeling
This is our implementation of Pointfilter, a network that automatically and robustly filters point clouds by removing noise and preserving their sharp features.

<p align="center"> <img src="Images/pipeline.png" width="75%"> </p>

The pipeline is built based on [PointNet](http://stanford.edu/~rqi/pointnet/) (a patch-based version of [PointNet](http://stanford.edu/~rqi/pointnet/)). Instead of using STN for alignment, we align the input patches by aligning their principal axes of the PCA with the Cartesian space.

## Environment
* Python 3.6
* PyTorch 1.0.0
* TensorboardX (1.6) if logging training info. 
* CUDA and CuDNN if training on the GPU (CUDA 9.0 & CuDNN 7.0)


## Datasets
You can download the datasets from the following [link](https://entuedu-my.sharepoint.com/:f:/g/personal/n1805982j_e_ntu_edu_sg/Er5PVpfMIBZDiucsZSUX-AsB8QXXHIfzVfENWSj1u9TNng?e=wEFDZY). Create a folder named Dataset and unzip the `Train.zip` and `Test.zip` files on it. In the datasets the input and ground truth point clouds are stored in different files with '.npy' extension. For each clean point cloud 'name.npy', there are 5 correponsing noisy models named as 'name_0.0025.npy', 'name_0.005.npy', 'name_0.01.npy', '0.015.npy', and 'name_0.025.npy'.  


## Setup
Install required python packages:
``` bash
pip install numpy
pip install scipy
pip install scikit-learn
pip install tensorboardX
```

Clone this repository:
``` bash
git clone https://github.com/dongbo-BUAA-VR/Pointfilter.git
cd Pointfilter
```

## Train
Use the script 'train.py' to train a model in the our dataset:
``` bash
cd Pointtilter
python train.py
```

## Test with Pre-trained Model
Use the script 'eval.py' to test your dataset:
``` bash
cd Pointtilter
python eval.py
```

## Acknowledgements
This code largely benefits from following repositories:
* [PointNet](http://stanford.edu/~rqi/pointnet/)
* [PCPNet](https://github.com/paulguerrero/pcpnet)
