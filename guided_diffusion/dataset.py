from pathlib import Path
from functools import partial
from collections import namedtuple

import torch
from torch import nn, einsum
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from torch.optim import Adam

from torchvision import transforms as T, utils

from einops import rearrange, reduce, repeat
from einops.layers.torch import Rearrange

from PIL import Image
from tqdm.auto import tqdm
from multiprocessing import cpu_count
# from ema_pytorch import EMA

# from accelerate import Accelerator

# import numpy as np
# from pytorch_fid.inception import InceptionV3
# from pytorch_fid.fid_score import calculate_frechet_distance

# from denoising_diffusion_pytorch.version import __version__

def exists(x):
    return x is not None

def convert_image_to_fn(img_type, image):
    if image.mode != img_type:
        return image.convert(img_type)
    return image

class Dataset(Dataset):
    def __init__(
        self,
        folder,
        con_folder,
        image_size,
        exts = ['jpg', 'jpeg', 'png', 'tiff'],
        cond_exits = False,
        augment_horizontal_flip = False,
        convert_image_to = None
    ):
        super().__init__()
        self.folder = folder
        self.con_folder = con_folder
        self.image_size = image_size
        self.input_paths = sorted([p for ext in exts for p in Path(f'{folder}').glob(f'**/*.{ext}')])
        self.con_paths = sorted([p for ext in exts for p in Path(f'{con_folder}').glob(f'**/*.{ext}')])

        self.cond_exits = cond_exits
        maybe_convert_fn = partial(convert_image_to_fn, convert_image_to) if exists(convert_image_to) else nn.Identity()

        self.transform = T.Compose([
            T.Lambda(maybe_convert_fn),
            T.Resize(image_size),
            T.RandomHorizontalFlip() if augment_horizontal_flip else nn.Identity(),
            T.CenterCrop(image_size),
            T.ToTensor()
        ])

    def __len__(self):
        return len(self.input_paths)

    def __getitem__(self, index):
        input_path = self.input_paths[index]
        con_path = self.con_paths[index]
        input_img = Image.open(input_path)
        con_img = Image.open(con_path)
        input_img = input_img.convert("L")
        if self.cond_exits:
            return self.transform(con_img), self.transform(input_img)
        else:
            return self.transform(con_img)
        
  
