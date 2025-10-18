import pickle
import os

data_path = os.listdir('/gdata1/lihl/NuScenes_processed_seq')
scenes = os.listdir(data_path)
scenes.sort()
assert(len(scenes) == 850)
        
scenes_train = scenes[:650] 
scenes_eval = scenes[-200:] 

with open('/gdata2/lihl/GenFlow3d/nuscenes_split/train.pkl', 'wb') as f:
    pickle.dump(scenes_train, f)
with open('/gdata2/lihl/GenFlow3d/nuscenes_split/eval.pkl', 'wb') as f:
    pickle.dump(scenes_eval, f)
