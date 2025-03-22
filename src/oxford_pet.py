import os
import torch
import shutil
import numpy as np

from PIL import Image
from tqdm import tqdm
from urllib.request import urlretrieve

import cv2
import random
import imutils


class OxfordPetDataset(torch.utils.data.Dataset):
    def __init__(self, root, mode="train", transform=None):

        assert mode in {"train", "valid", "test"}

        self.root = root
        self.mode = mode
        self.transform = transform

        self.images_directory = os.path.join(self.root, "images")
        self.masks_directory = os.path.join(self.root, "annotations", "trimaps")
        
        if not (os.path.exists(self.images_directory) and os.path.exists(self.masks_directory)):
            self.download(self.root)
            
        self.filenames = self._read_split()  # read train/valid/test splits

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        filename = self.filenames[idx]
        image_path = os.path.join(self.images_directory, filename + ".jpg")
        mask_path = os.path.join(self.masks_directory, filename + ".png")

        image = np.array(Image.open(image_path).convert("RGB"))

        trimap = np.array(Image.open(mask_path))
        mask = self._preprocess_mask(trimap)

        
        sample = dict(image=image, mask=mask, mode= self.mode, filename = filename)  # , trimap=trimap
        if self.transform is not None:
            sample = self.transform(**sample)

        return sample

    @staticmethod
    def _preprocess_mask(mask):
        mask = mask.astype(np.float32)
        mask[mask == 2.0] = 0.0
        mask[(mask == 1.0) | (mask == 3.0)] = 1.0
        return mask

    def _read_split(self):
        split_filename = "test.txt" if self.mode == "test" else "trainval.txt"
        split_filepath = os.path.join(self.root, "annotations", split_filename)
        with open(split_filepath) as f:
            split_data = f.read().strip("\n").split("\n")
        filenames = [x.split(" ")[0] for x in split_data]
        if self.mode == "train":  # 90% for train
            filenames = [x for i, x in enumerate(filenames) if i % 10 != 0]
        elif self.mode == "valid":  # 10% for validation
            filenames = [x for i, x in enumerate(filenames) if i % 10 == 0]
        return filenames

    @staticmethod
    def download(root):

        # load images
        filepath = os.path.join(root, "images.tar.gz")
        download_url(
            url="https://www.robots.ox.ac.uk/~vgg/data/pets/data/images.tar.gz",
            filepath=filepath,
        )
        extract_archive(filepath)

        # load annotations
        filepath = os.path.join(root, "annotations.tar.gz")
        download_url(
            url="https://www.robots.ox.ac.uk/~vgg/data/pets/data/annotations.tar.gz",
            filepath=filepath,
        )
        extract_archive(filepath)


class SimpleOxfordPetDataset(OxfordPetDataset):
    def __getitem__(self, *args, **kwargs):

        sample = super().__getitem__(*args, **kwargs)

        # resize images
        image = np.array(Image.fromarray(sample["image"]).resize((256, 256), Image.BILINEAR))
        mask = np.array(Image.fromarray(sample["mask"]).resize((256, 256), Image.NEAREST))
        # trimap = np.array(Image.fromarray(sample["trimap"]).resize((256, 256), Image.NEAREST))

        # convert to other format HWC -> CHW
        sample["image"] = (np.moveaxis(image, -1, 0)/255.0).astype(np.float32)
        sample["mask"] = (mask.flatten()).astype(np.float32)
        # sample["trimap"] = np.expand_dims(trimap, 0)

        return sample


class TqdmUpTo(tqdm):
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)


def download_url(url, filepath):
    directory = os.path.dirname(os.path.abspath(filepath))
    os.makedirs(directory, exist_ok=True)
    if os.path.exists(filepath):
        return

    with TqdmUpTo(
        unit="B",
        unit_scale=True,
        unit_divisor=1024,
        miniters=1,
        desc=os.path.basename(filepath),
    ) as t:
        urlretrieve(url, filename=filepath, reporthook=t.update_to, data=None)
        t.total = t.n


def extract_archive(filepath):
    extract_dir = os.path.dirname(os.path.abspath(filepath))
    dst_dir = os.path.splitext(filepath)[0]
    if not os.path.exists(dst_dir):
        shutil.unpack_archive(filepath, extract_dir)
                
                
def transform(image, mask, mode, filename):
    # only augment on training set
    if mode == "train":        
        # random flip
        if random.random()>0.5:
            image = cv2.flip(image, 1)
            mask = cv2.flip(mask, 1)
            #trimap = cv2.flip(trimap, 1)
        
        # random rotate    
        angel = random.uniform(-15,15)
        image = imutils.rotate(image, angel)
        mask = imutils.rotate(mask, angel)
        #trimap = imutils.rotate(trimap, angel)
        
        # # random scale 
        # scale = random.uniform(0.9, 1.1)
        # new_w = int(image.shape[1] * scale)
        # new_h = int(image.shape[0] * scale)
        # image = cv2.resize(image, (new_w, new_h))
        # mask = cv2.resize(mask, (new_w, new_h))
        # #trimap = cv2.resize(trimap, (new_w, new_h))
        
        # random translation (left-right shift)
        max_shift = 10 
        shift_x = random.randint(-max_shift, max_shift) 
        shift_y = random.randint(-max_shift, max_shift) 
        M = np.float32([[1, 0, shift_x], [0, 1, shift_y]])  
        image = cv2.warpAffine(image, M, (image.shape[1], image.shape[0]))
        mask = cv2.warpAffine(mask, M, (mask.shape[1], mask.shape[0]))
        #trimap = cv2.warpAffine(trimap, M, (trimap.shape[1], trimap.shape[0]))
        
        # Gaussian noise
        mean = 0 
        stddev = 0.01 
        noise = np.random.normal(mean, stddev, image.shape).astype(np.uint8)
        image = cv2.add(image, noise)         
    
    return dict(image=image, mask=mask)


def load_dataset(data_path, mode = "train"):
    # implement the load dataset function here
    if mode == "train":
        return SimpleOxfordPetDataset(root = data_path, mode = mode, transform = transform)
    
    elif mode == "valid":
        return SimpleOxfordPetDataset(root = data_path, mode = mode)
    
    elif mode == "test":
        return SimpleOxfordPetDataset(root = data_path, mode = mode)
        
    else:
        raise ValueError( f"Invalid mode: {mode}")