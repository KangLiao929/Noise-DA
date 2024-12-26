import torch.utils.data as data
import torch
import numpy as np
import random
import scipy.io as sio
from core.dataset_utils import random_crop, make_dataset, load_img, Augment_RGB_torch

class DatasetTrain(data.Dataset):
    def __init__(self, data_root, gt_root, real_root, ref_root, data_len=-1, patch_size=128, denoise=False, loader=load_img, norm="norm0"):
        syn_path = make_dataset(data_root)      # synthetic data    
        gt_path = make_dataset(gt_root)         # synthetic gt data        
        real_path = make_dataset(real_root)     # real-world data (w/o gt)
        ref_path = make_dataset(ref_root)       # reference data (from paired synthetic gt data or large-scale dataset)

        if data_len > 0:
            self.syn_path = syn_path[:int(data_len)]
            self.gt_path = gt_path[:int(data_len)]
            self.real_path = real_path[:int(data_len)]
            self.ref_path = ref_path[:int(data_len)]
        else:
            self.syn_path = syn_path
            self.gt_path = gt_path
            self.real_path = real_path
            self.ref_path = ref_path

        self.loader = loader
        self.norm = norm
        self.ps = patch_size
        self.denoise = denoise
        self.augment = Augment_RGB_torch()
        self.transforms_aug = [method for method in dir(self.augment) if callable(getattr(self.augment, method)) if not method.startswith('_')]
        self.sigma = [0, 75]
        self.sigma_min, self.sigma_max = self.sigma[0], self.sigma[1]

    def shuffle_paths(self):
        '''randomly shuffle the index of real images and reference images in each epoch'''
        random.shuffle(self.real_path)
        random.shuffle(self.ref_path)
    
    def __getitem__(self, index):
        ret = {}
        syn_path = self.syn_path[index % len(self.syn_path)]
        gt_path = self.gt_path[index % len(self.gt_path)]
        real_path = self.real_path[index % len(self.real_path)]
        ref_path = self.ref_path[index % len(self.ref_path)]

        syn_img = self.loader(syn_path, self.norm)
        gt_img = self.loader(gt_path, self.norm)
        real_img = self.loader(real_path, self.norm)
        ref_img = self.loader(ref_path, self.norm)
        
        syn_img = torch.from_numpy(np.float32(syn_img))
        syn_img = syn_img.permute(2,0,1)
        gt_img = torch.from_numpy(np.float32(gt_img))
        gt_img = gt_img.permute(2,0,1)
        real_img = torch.from_numpy(np.float32(real_img))
        real_img = real_img.permute(2,0,1)
        ref_img = torch.from_numpy(np.float32(ref_img))
        ref_img = ref_img.permute(2,0,1)
        
        '''randomly crop the image given the patch size'''
        syn_img, gt_img, real_img, ref_img = random_crop(syn_img, gt_img, real_img, ref_img, self.ps)
            
        '''randomly rotate and clip the image'''
        apply_trans = self.transforms_aug[random.getrandbits(3)] # get rand index from [0, 2^3]
        apply_trans_real = self.transforms_aug[random.getrandbits(3)]
        apply_trans_large = self.transforms_aug[random.getrandbits(3)]
        syn_img = getattr(self.augment, apply_trans)(syn_img)
        gt_img = getattr(self.augment, apply_trans)(gt_img)
        real_img = getattr(self.augment, apply_trans_real)(real_img)
        ref_img = getattr(self.augment, apply_trans_large)(ref_img)
        
        '''add random gaussian noise for blind denoising task'''
        if(self.denoise):
            noise_level = torch.FloatTensor([np.random.uniform(self.sigma_min, self.sigma_max)])/255.0
            noise = torch.randn(syn_img.size()).mul_(noise_level).float()
            syn_img.add_(noise)
        
        ret['gt_img'] = gt_img
        ret['syn_img'] = syn_img
        ret['real_img'] = real_img
        ret['ref_img'] = ref_img
        ret['path'] = syn_path.rsplit("/")[-1].rsplit("\\")[-1]
        return ret

    def __len__(self):
        return len(self.syn_path)
    
class DatasetTest(data.Dataset):
    def __init__(self, data_root, gt_root="", flag_label=False, real_denoise=False, loader=load_img, norm="norm0"):
        self.flag_label = flag_label
        self.input_path = make_dataset(data_root)
        if(self.flag_label):
            self.gt_path = make_dataset(gt_root)
        self.loader = loader
        self.real_denoise = real_denoise
        self.norm = norm

    def __getitem__(self, index):
        if(self.real_denoise):
            ret = {}
            input_path = self.input_path[index]
            img = sio.loadmat(input_path)
            Inoisy = np.float32(np.array(img['ValidationNoisyBlocksSrgb']))
            Inoisy /= 255.
            restored = np.zeros_like(Inoisy)
            
            ret['input_img'] = Inoisy
            ret['gt_img'] = restored
            ret['path'] = ""
            return ret
        else:
            ret = {}
            input_path = self.input_path[index]
            if(self.flag_label):
                input_img = self.loader(input_path, self.norm)
                gt_path = self.gt_path[index]
                gt_img = self.loader(gt_path, self.norm)
            else:
                input_img = self.loader(input_path, self.norm)
                gt_img = input_img
            
            input_img = torch.from_numpy(np.float32(input_img))
            input_img = input_img.permute(2,0,1)
            gt_img = torch.from_numpy(np.float32(gt_img))
            gt_img = gt_img.permute(2,0,1)
            
            ret['input_img'] = input_img
            ret['gt_img'] = gt_img
            ret['path'] = input_path.rsplit("/")[-1].rsplit("\\")[-1]
            return ret

    def __len__(self):
        return len(self.input_path)