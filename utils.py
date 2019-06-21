import cv2
import numpy as np
from skimage.transform import resize
from torch.utils.data import Dataset, DataLoader
import os
import re
import torch
import h5py
from .model import ESPNet, Embedding
import torch.functional as F


def crop_im_by_circle(img):
    downscale = 128
    h, w = img.shape[:2]
    im = cv2.resize(img, (downscale, downscale), interpolation=cv2.INTER_NEAREST)
    mask = np.zeros(im.shape[:2], dtype='uint8')
    bgdModel = np.zeros((1, 65))
    fgdModel = np.zeros((1, 65))
    rect = (1, 1, downscale-2, downscale-2)
    cv2.grabCut(im,mask,rect,bgdModel,fgdModel,5,cv2.GC_INIT_WITH_RECT)
    kernel = np.ones((5, 5),np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask[32:96, 32:96] = 1
    cv2.grabCut(im,mask,None,bgdModel,fgdModel,5,cv2.GC_INIT_WITH_MASK)
    left, up, width, height = cv2.boundingRect(mask%2)
    left = int(left*w/128)
    up = int(up*h/128)
    width = int(width*w/128)
    height = int(height*h/128)
    if width > 0 and height > 0:
        return cv2.resize(img[up: up+height, left:left+width, :], (512, 512))
    return


def im_resize(im, size, anti_aliasing):
    w, h = size
    return resize(im.astype('d'), (w, h, im.shape[2]), mode='constant', anti_aliasing=anti_aliasing).astype('f')


class MyDataset(Dataset):
    def __init__(self, dataset_path, regex_ext, psp=None):  # psp:= pattern specific processing, (pattern, function)
        """
        pattern_dict contains "img", "mask" and other data attributes as keys,
        and has correspoinding regular expression pattern and
        processing function as values.
        """
        self.files = []
        self.dataset_path = dataset_path
        self.psp = psp
        for f in sorted(os.listdir(dataset_path)):
            if re.match(regex_ext, f):
                self.files.append(f)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        img = cv2.imread(self.dataset_path.rstrip('/') + '/' + self.files[idx])
        img = crop_im_by_circle(img)
        img = im_resize(img, (512, 512), 1)
        if self.psp:
            pattern, func = self.psp
            img = func(re.match(pattern, self.files[idx]).group(1), img)
        img -= np.array([0.485, 0.456, 0.406]).reshape(1, 1, 3)
        img /= np.array([0.229, 0.224, 0.225]).reshape(1, 1, 3)
        return (torch.from_numpy(img.transpose(2, 0, 1)),
                self.files[idx])


def compute(directory, re_pattern, out_file, seg_model=ESPNet(), seg_model_wts="r_embedding/esp_model_wts.pt",
               emb_model=Embedding(128), emb_model_wts="r_embedding/model_wts.pth",
               device="cuda:0", batch_size=32):
    """
    precompute the segmentation mask using ESPNet
    """
    with h5py.File(out_file, 'w') as f:
        seg_model = seg_model.to(device).eval()
        seg_model.load_state_dict(torch.load(seg_model_wts))
        emb_model = emb_model.to(device).eval()
        emb_model.load_state_dict(torch.load(emb_model_wts))
        d = MyDataset(directory, re_pattern)
        f.create_dataset('embs', (len(d), 128), dtype='f')
        cnt = 0
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1).to(device)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1).to(device)
        flist = []
        for ims, f_names in DataLoader(d, batch_size=batch_size):
            flist.extend(f_names)
            with torch.no_grad():
                segs = F.softmax(seg_model(ims), 1)[:, :1, ...]
                segs -= mean
                segs /= std
                embs = emb_model(segs).cpu().numpy()
                f['embs'][cnt * batch_size:(cnt + 1) * batch_size, :] = embs
            cnt += 1
            print(cnt * batch_size)
        f['embs'].attrs['file_name'] = flist

