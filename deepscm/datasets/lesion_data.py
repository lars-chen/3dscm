import numpy as np
import pandas as pd
import torch
import nibabel as nib
from  scipy import ndimage
from torch.utils.data.dataset import Dataset

def resize_data_volume_by_scale(data, scale):
   """
   Resize the data based on the provided scale
   """
   scale_list = [scale,scale,scale]
   return ndimage.interpolation.zoom(data, scale_list, order=0)

class  NvidiaDataset(Dataset):
    def __init__(self, data_dir = "../../lesion_mri", train=True):
        super().__init__()

        self.data_dir = data_dir
        
        subjects = pd.read_csv(self.data_dir + "/participants_lesioned.csv")
        subjects['ventricle_vol'] = subjects['ventricle_vol']/1e3 # convert to ml
        subjects['brain_vol'] = subjects['brain_vol']/1e3
        subjects['lesion_vol'] = subjects['true_lesion_vol']/1e3

        
        if train:
            self.subjects = subjects[:500] # TODO: change to 4500
        else:
            self.subjects = subjects[4800:].reset_index() # TODO change to 4500
        
    def __len__(self):
        return len(self.subjects)
    
    def __getitem__(self, index):
        # load labels
        item = dict()
        item["age"] = self.subjects["age"][index]
        item["score"] = self.subjects["score"][index]
        item["sex"] = self.subjects["sex"][index]
        item["brain_volume"] = self.subjects["brain_vol"][index]  
        item["ventricle_volume"] = self.subjects["ventricle_vol"][index]
        item["num_lesions"] = self.subjects["lesion_num"][index] 
        item["lesion_volume"] = self.subjects["true_lesion_vol"][index] + 1e-5
        
        # process centers into indexs        
        centers = self.subjects["centers"][index]
        # for char in ' ][)(array':
        #     centers = centers.replace(char, '')
        # centers = centers.split(',')
        # if centers != ['']:
        #     centers = np.array([int(str) for str in centers]).reshape((-1, 3))
        # else:
        #     centers = np.array([])
        # print(centers)
        item["centers"] = centers

        # load image
        participant_id = str(self.subjects["subject"][index])
        participant_id = '00000'[:5-len(participant_id)] + participant_id
        img_dir = f"{self.data_dir}/{participant_id}.nii.gz"    
        img = nib.load(img_dir).get_fdata()[12:148, 8:212, :136] #* 255 #8:212
        img = resize_data_volume_by_scale(img, 0.47)[np.newaxis, :, :, :] #0.94
        item["image"] = np.clip(img, 0, 1.5)*255/1.5 # TODO: map to 255?
        return item
