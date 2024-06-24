import numpy as np
import pandas as pd
import h5py
from PIL import Image
import torch
from torch.utils.data.dataset import Dataset
import torchvision as tv


class CISDataset(Dataset):
#    def __init__(self, csv_path, crop_type=None, crop_size=(192, 192), resize=None, eps:float=1e-4):
    def __init__(self, h5_path, type, crop_type=None, crop_size=(182, 182), resize=None, eps:float=1e-4, downsample: int = None):
        super().__init__()
        
        if type == 'train':
            remove = [193, 438, 228, 297, 373, 387, 402, 405] #193 excluded because looks weird. Others contain NaN
        elif type == 'val':
            remove = [31, 116, 119, 78, 79, 80, 81, 82, 83, 84]
        else: 
            remove = [22]

        h5 = h5py.File(h5_path,'r')
        
        img = np.delete(h5.get('flair')[:].transpose(0,2,1), remove, 0)
        img = (img - img.min())/(img.max() - img.min())
        mean = img.mean()
        img[img < mean] = 0
        self.mri = img


        self.subject = np.delete(h5.get('subject')[:], remove, 0)
        self.duration = np.delete(h5.get('duration')[:], remove, 0) + eps
        self.diagnosis = np.delete(h5.get('diagnosis')[:], remove, 0)
        self.sex = np.delete(h5.get('sex')[:], remove, 0)
        self.age= np.delete(h5.get('age')[:], remove, 0)
        self.edss = np.delete(h5.get('edss')[:], remove, 0) + eps
        self.relapse = np.delete(h5.get('relapse')[:], remove, 0)
        self.treatment = np.delete(h5.get('treatment')[:], remove, 0) 
        self.treatment_propagated = np.delete(h5.get('treatment_propagated')[:], remove, 0)
        self.brain_vol = np.delete(h5.get('brain_volume')[:], remove, 0) + eps
        self.lesion_vol = np.delete(h5.get('lesion_volume')[:], remove, 0) + eps
        self.lesion_count = np.delete(h5.get('lesion_count')[:], remove, 0)
        h5.close()
        
        self.eps = eps
        self.downsample = downsample
        self.crop_type = crop_type
        self.crop_size = crop_size
        self.resize = resize

    def __len__(self):
        return len(self.subject)

    def __getitem__(self, index):

        item = dict()
        item['age'] = self.age[index]
        item['sex'] = self.sex[index] ### 1: male, 0: female
        item['type'] = self.diagnosis[index] ### 1: RRMS, 0: CIS
        item['relapse'] = self.relapse[index]
        item['duration'] = np.nan_to_num(self.duration[index])
        item['brain_vol'] = np.nan_to_num(self.brain_vol[index]).astype(np.float32)
        item['lesion_vol'] = np.nan_to_num(self.lesion_vol[index]).astype(np.float32)
        item['edss'] = np.nan_to_num(self.edss[index]) 
        item['treatment'] = self.treatment[index]   
        '''np.nan: 0, 'none': 0., 'Glatirameracetat (Copaxone®)': 1.,
        'Interferon beta (Avonex®, Betaferon®, Extavia®, Rebif®)': 2., 
        'Fumarat (Tecfidera®)': 3., 
        'sonstige': 4., 
        'Natalizumab (Tysabri®)': 5., 
        'Fingolimod (Gilenya®) ': 6., 
        'Teriflunomid (Aubagio®)': 7., 
        'Alemtuzumab (Lemtrada®) ': 8.})'''
        
        item['treatment_propagated'] = self.treatment_propagated[index]
        img = self.mri[index]

        transform_list = []
        if self.crop_type is not None:
            if self.crop_type == 'center':
                transform_list += [tv.transforms.CenterCrop(self.crop_size)]
            elif self.crop_type == 'random':
                transform_list += [tv.transforms.RandomCrop(self.crop_size)]
            else:
                raise ValueError(f'unknown crop type: {self.crop_type}')

        if self.downsample is not None and self.downsample > 1:   # LPC
            transform_list += [tv.transforms.Resize(tuple(np.array(self.crop_size) // self.downsample), interpolation=3)] # LPC

        transform_list += [tv.transforms.ToTensor()]
        img = tv.transforms.Compose(transform_list)(Image.fromarray(img))
        item['image'] = img
        return item

    @staticmethod
    def _prepare_item(item):
        eps = 1e-10
        item['age'] = torch.as_tensor(item['age'], dtype=torch.float) 
        item['sex'] = torch.as_tensor(item['type'], dtype=torch.float)
        item['type'] = torch.as_tensor(item['type'], dtype=torch.float)
        item['relapse'] = torch.as_tensor(item['relapse'], dtype=torch.float)
        item['duration'] = torch.as_tensor(item['duration'], dtype=torch.float)
        item['slice_brain_volume'] = torch.as_tensor(item['slice_brain_volume'], dtype=torch.float)
        item['slice_ventricle_volume'] = torch.as_tensor(item['slice_ventricle_volume'], dtype=torch.float)
        item['slice_lesion_volume'] = torch.as_tensor(item['slice_lesion_volume'], dtype=torch.float)
        item['brain_vol'] = torch.as_tensor(item['brain_vol'], dtype=torch.float) + eps
        item['ventricle_volume'] = torch.as_tensor(item['ventricle_volume'], dtype=torch.float) + eps
        item['lesion_vol'] = torch.as_tensor(item['lesion_vol'], dtype=torch.float) + eps
        item['edss'] = torch.as_tensor(item['edss'], dtype=torch.float) 
        item['fss'] = torch.as_tensor(item['fss'], dtype=torch.float)
        item['msss'] = torch.as_tensor(item['msss'], dtype=torch.float)
        item['treatment'] = torch.as_tensor(item['treatment'], dtype=torch.float)
        item['treatment_propagated'] = torch.as_tensor(item['treatment_propagated'], dtype=torch.float)
        item['slice_number'] = torch.as_tensor(item['slice_number'], dtype=torch.float)
        return item
