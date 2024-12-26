import torch
import os
import torch
import numpy as np
import cv2
import torch.nn.functional as F
import re

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)

def get_first_number(path):
    match = re.search(r'\d+', path)
    return int(match.group()) if match else -1

def make_dataset(dir):
    if os.path.isfile(dir):
        images = [i for i in np.genfromtxt(dir, dtype=str, encoding='utf-8')]
    else:
        images = []
        assert os.path.isdir(dir), '%s is not a valid directory' % dir
        for root, _, fnames in sorted(os.walk(dir)):
            for fname in sorted(fnames):
                path = os.path.join(root, fname)
                images.append(path)

    images = sorted(images, key=get_first_number)
    return images           

def load_img(filepath, norm="norm0"):
    img = cv2.imread(filepath)
    if len(img.shape) == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    elif img.shape[2] == 1:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    else:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32)
    img = img/255. if norm == "norm0" else img/127.5-1.
    return img

def random_crop(syn_img, gt_img, real_img, ref_img, ps):
    """
    Randomly crop or resize images to the given patch size.
    Ensures syn_img and gt_img are cropped in the same region.
    """
    def crop_or_resize(img, r=None, c=None):
        h, w = img.shape[1:]
        if h > ps and w > ps and r is None and c is None:
            r, c = np.random.randint(0, h - ps), np.random.randint(0, w - ps)
        return img[:, r:r + ps, c:c + ps] if r is not None and c is not None else F.interpolate(img.unsqueeze(0), size=(ps, ps), mode='bilinear', align_corners=False).squeeze(0)

    # Crop syn_img and gt_img with shared region
    r, c = np.random.randint(0, syn_img.shape[1] - ps), np.random.randint(0, syn_img.shape[2] - ps) if syn_img.shape[1] > ps and syn_img.shape[2] > ps else (None, None)
    syn_img, gt_img = crop_or_resize(syn_img, r, c), crop_or_resize(gt_img, r, c)

    # Independently process real_img and ref_img
    real_img, ref_img = crop_or_resize(real_img), crop_or_resize(ref_img)

    return syn_img, gt_img, real_img, ref_img

class Augment_RGB_torch:
    '''randomly rotate and flip the image'''
    def __init__(self):
        pass
    def transform0(self, torch_tensor):
        return torch_tensor   
    def transform1(self, torch_tensor):
        torch_tensor = torch.rot90(torch_tensor, k=1, dims=[-1,-2])
        return torch_tensor
    def transform2(self, torch_tensor):
        torch_tensor = torch.rot90(torch_tensor, k=2, dims=[-1,-2])
        return torch_tensor
    def transform3(self, torch_tensor):
        torch_tensor = torch.rot90(torch_tensor, k=3, dims=[-1,-2])
        return torch_tensor
    def transform4(self, torch_tensor):
        torch_tensor = torch_tensor.flip(-2)
        return torch_tensor
    def transform5(self, torch_tensor):
        torch_tensor = (torch.rot90(torch_tensor, k=1, dims=[-1,-2])).flip(-2)
        return torch_tensor
    def transform6(self, torch_tensor):
        torch_tensor = (torch.rot90(torch_tensor, k=2, dims=[-1,-2])).flip(-2)
        return torch_tensor
    def transform7(self, torch_tensor):
        torch_tensor = (torch.rot90(torch_tensor, k=3, dims=[-1,-2])).flip(-2)
        return torch_tensor