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
                       |
                       |--samples
                       |
                       |--sweeps
                       |
                       |--v1.0-trainval
                       |
                       |--LICENSE
```

Then, run the code file `./data/nuscenes/create_nuscenes_seq_length.py`:
```
cd ./data/nuscenes
python create_nuscenes_seq_length.py --nusc_root your_path_to_nuscenes --path_out OUTPUT_PATH
cd ../..
```  
The processed nuScenes dataset is in `OUTPUT_PATH`


