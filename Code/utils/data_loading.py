# import logging
from os import listdir
from os.path import splitext
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageOps
from torch.utils.data import Dataset

from skimage.util.shape import view_as_windows
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img

class BasicDataset(Dataset):
    def __init__(self, images_dir: str, masks_dir: str, scale: list = [1000,1000], mask_suffix: str = '_bin_mask'):
        self.images_dir = Path(images_dir)
        self.masks_dir = Path(masks_dir)
        # assert 0 < scale <= 1, 'Scale must be between 0 and 1'
        self.scale = scale
        self.mask_suffix = mask_suffix

        self.ids = [splitext(file)[0] for file in listdir(images_dir) if not file.startswith('.')]
        if not self.ids:
            raise RuntimeError(f'No input file found in {images_dir}, make sure you put your images there')
        # logging.info(f'Creating dataset with {len(self.ids)} examples')

    def __len__(self):
        return len(self.ids)

    @classmethod
    def preprocess(cls, pil_img, scale, is_mask):
        # w, h = pil_img.size
        newW = scale[0]
        newH = scale[1]
        # newW, newH = int(scale * w), int(scale * h)
        assert newW > 0 and newH > 0, 'Scale is too small, resized images would have no pixel'
        if is_mask:
            pil_img = ImageOps.grayscale(pil_img)

        # pil_img = pil_img.resize((newW, newH))
        # img_ndarray = np.asarray(pil_img)
        img_ndarray = img_to_array(pil_img)

        # if img_ndarray.ndim == 2 and not is_mask:
        #     img_ndarray = img_ndarray[np.newaxis, ...]
        # elif not is_mask:
        #     img_ndarray = img_ndarray.transpose((2, 0, 1))

        # if not is_mask:
        img_ndarray = img_ndarray / 255.0
        patch_width = 256
        patch_height = 256
        imgs = []
        if not is_mask:
            new_imgs = view_as_windows(img_ndarray, (patch_width, patch_height, 3), (patch_width//2, patch_height//2, 3))
        else:
            new_imgs = view_as_windows(img_ndarray, (patch_width, patch_height, 1), (patch_width//2, patch_height//2, 1))
        
        for im in new_imgs:
            imgs.append(im)
        if not is_mask:
            imgs = imgs.reshape(-1,patch_height,patch_width,3)
        else:
            imgs = imgs.reshape(-1,patch_height,patch_width,1)
        return imgs#img_ndarray

    @classmethod
    def load(cls, filename, scale):
        ext = splitext(filename)[1]
        if ext in ['.npz', '.npy']:
            return Image.fromarray(np.load(filename))
        elif ext in ['.pt', '.pth']:
            return Image.fromarray(torch.load(filename).numpy())
        else:
            return load_img(filename, color_mode='rgb',target_size=scale)
            # return Image.open(filename)

    def __getitem__(self, idx):
        name = self.ids[idx]
        mask_file = list(self.masks_dir.glob(name + self.mask_suffix + '.*'))
        img_file = list(self.images_dir.glob(name + '.*'))

        assert len(mask_file) == 1, f'Either no mask or multiple masks found for the ID {name}: {mask_file}'
        assert len(img_file) == 1, f'Either no image or multiple images found for the ID {name}: {img_file}'
        mask = self.load(mask_file[0],self.scale)
        img = self.load(img_file[0],self.scale)

        assert img.size == mask.size, \
            'Image and mask {name} should be the same size, but are {img.size} and {mask.size}'

        img = self.preprocess(img, self.scale, is_mask=False)
        mask = self.preprocess(mask, self.scale, is_mask=True)

        return {
            'image': torch.as_tensor(img.copy()).float().contiguous(),
            'mask': torch.as_tensor(mask.copy()).long().contiguous()
        }


class CarvanaDataset(BasicDataset):
    def __init__(self, images_dir, masks_dir, scale=1):
        super().__init__(images_dir, masks_dir, scale, mask_suffix='_mask')