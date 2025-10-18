# GenFlow3D
**[ICCV 2025] Code for "GenFlow3D: Generative scene flow estimation and prediction on point cloud sequences".**
**Hanlin Li, Wenming Weng, Yueyi Zhang†, Zhiwei Xiong**

## Preproration
To run our model, install the PointNet2 library first as follows.
```
cd pointnet2
python setup.py install
cd ..
```
After this, we need to prepare the sequential scene flow datasets nuScenes and Argoverse 2.

### nuScenes
The code to preprocess the nuScenes dataset is in `./data/nuscenes`. We follow [SLIM](https://github.com/mercedes-benz/selfsupervised_flow) to generate the scene flow ground truth.

Download the full nuScenes v1.0 trainval dataset from the [official website](https://www.nuscenes.org/nuscenes#download) and unzip them. The directory structure should be as follows.
```
your_path_to_nuscenes--|
                       |--maps
                       |--samples
                       |--sweeps
                       |--v1.0-trainval
                       |--LICENSE
```

Then, run the code file `./data/nuscenes/create_nuscenes_seq_length.py`:
```
cd ./data/nuscenes
python create_nuscenes_seq_length.py --nusc_root your_path_to_nuscenes --path_out OUTPUT_PATH_NUSCENES
cd ../..
```  
The processed nuScenes dataset is in `OUTPUT_PATH_NUSCENES`.

### Argoverse 2
The code to preprocess the Argoverse 2 dataset is in `./data/av2`. We use the official method from [av2-api](https://github.com/argoverse/av2-api) to generate the scene flow groundtruth.

Download the training parts and validation parts of the Argoverse 2 Sensor Dataset from the [official website](https://www.argoverse.org/av2.html#download-link) and unzip them. The directory structure should be as follows.
```
your_path_to_av2--|
                  |--av2--|
                          |--sensor--|
                                     |--train--|
                                     |         |--00a6ffc1-6ce9-3bc3-a060-6006e9893a1a
                                     |         |--... 
                                     |--val--|
                                     |       |--0aa4e8f5-2f9a-39a1-8f80-c2fdde4405a2
                                     |       |--...
```

Then run the code file `./data/av2/create_av2_seq.py`：
```
cd ./data/av2
python create_av2_seq.py --data_root your_path_to_av2 --out_root OUTPUT_PATH_AV2 --split_name train
python create_av2_seq.py --data_root your_path_to_av2 --out_root OUTPUT_PATH_AV2 --split_name val
```
The processed nuScenes dataset is in `OUTPUT_PATH_AV2`.
