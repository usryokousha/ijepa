# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from logging import getLogger

from PIL import ImageFilter

import torch
import PIL.Image as Image
import torchvision.transforms as transforms
from torchvision.transforms import functional as F

_GLOBAL_SEED = 0
logger = getLogger()


def make_transforms(
    crop_size=224,
    crop_scale=(0.3, 1.0),
    color_jitter=1.0,
    horizontal_flip=False,
    color_distortion=False,
    gaussian_blur=False,
    normalization=((0.485, 0.456, 0.406),
                   (0.229, 0.224, 0.225))
):
    logger.info('making imagenet data transforms')

    def get_color_distortion(s=1.0):
        # s is the strength of color distortion.
        color_jitter = transforms.ColorJitter(0.8*s, 0.8*s, 0.8*s, 0.2*s)
        rnd_color_jitter = transforms.RandomApply([color_jitter], p=0.8)
        rnd_gray = transforms.RandomGrayscale(p=0.2)
        color_distort = transforms.Compose([
            rnd_color_jitter,
            rnd_gray])
        return color_distort

    transform_list = []
    transform_list += [transforms.RandomResizedCrop(crop_size, scale=crop_scale)]
    if horizontal_flip:
        transform_list += [transforms.RandomHorizontalFlip()]
    if color_distortion:
        transform_list += [get_color_distortion(s=color_jitter)]
    if gaussian_blur:
        transform_list += [GaussianBlur(p=0.5)]
    transform_list += [transforms.ToTensor()]
    transform_list += [transforms.Normalize(normalization[0], normalization[1])]

    transform = transforms.Compose(transform_list)
    return transform

class RandomTransformWrapper:
    def __init__(self, transforms_list):
        if not isinstance(transforms_list, list):
            self.transforms = [transforms_list]
        else:
            self.transforms = transforms_list
        self._is_random = any(["random" in t.__class__.__name__.lower() for t in transforms_list])

    def __call__(self, *images):
        transform_idx = [0] * len(images) if len(self.transforms) == 1 else range(len(images))
        if not self._is_random:
            transformed_images = [self.transforms[i](img) for i, img in zip(transform_idx, images)]
            
        torch_state = torch.random.get_rng_state()
        if torch.cuda.is_available():
            torch_cuda_state = torch.cuda.get_rng_state_all()
        else:
            torch_cuda_state = None

        transformed_images = []
        for i, img in zip(transform_idx, images):
            torch.random.set_rng_state(torch_state)
            if torch.cuda.is_available():
                torch.cuda.set_rng_state_all(torch_cuda_state) # pyright: ignore
            transformed_images.append(self.transforms[i](img))

        return tuple(transformed_images)

class CrossTransform(object):
    def __init__(
            self, 
            crop_size=96,
            crop_scale=(0.3, 1.0),
            latent_crop_size=224,
            latent_crop_scale=(0.3, 1.0),
            color_jitter=1.0,
            horizontal_flip=False,
            color_distortion=False,
            gaussian_blur=False,
            normalization=((0.485, 0.456, 0.406),
                            (0.229, 0.224, 0.225))):
        def get_color_distortion(s=1.0, p=0.2):
            # s is the strength of color distortion.
            color_jitter = transforms.ColorJitter(0.8*s, 0.8*s, 0.8*s, 0.2*s)
            rnd_color_jitter = transforms.RandomApply([color_jitter], p=0.8)
            rnd_gray = transforms.RandomGrayscale(p=p)
            color_distort = transforms.Compose([
                rnd_color_jitter,
                rnd_gray])
            return color_distort

        self.downsample = Downsample()

        # common transforms
        transform_list = [transforms.RandomResizedCrop(crop_size, scale=crop_scale),
                          transforms.RandomResizedCrop(latent_crop_size, scale=latent_crop_scale),]
        
        self.common_transform = RandomTransformWrapper(transform_list)

        # vision transforms
        vision_transform_list = []
        if horizontal_flip:
            vision_transform_list += [transforms.RandomHorizontalFlip()]
        if color_distortion:
            vision_transform_list += [get_color_distortion(s=color_jitter, p=1.0)]
        vision_transform_list += [transforms.ToTensor()]
        vision_transform_list += [transforms.Normalize(normalization[0], normalization[1])]

        self.vision_transform = transforms.Compose(vision_transform_list)

        # latent transforms (for cross-modal prediction)
        latent_transform_list = []
        if horizontal_flip:
            latent_transform_list += [transforms.RandomHorizontalFlip()]
        if color_distortion:
            latent_transform_list += [get_color_distortion(s=color_jitter)]
        if gaussian_blur:
            latent_transform_list += [GaussianBlur(p=0.5)]
        latent_transform_list += [transforms.ToTensor()]
        latent_transform_list += [transforms.Normalize(normalization[0], normalization[1])]

        self.latent_transform = transforms.Compose(latent_transform_list)

    def __call__(self, img):
        img_latent = img.copy()
        img = self.downsample(img)
        # common transforms
        img, img_latent = self.common_transform(img, img_latent)
        # vision transforms
        img = self.vision_transform(img)
        # latent transforms
        img_latent = self.latent_transform(img_latent)
        return img, img_latent



class GaussianBlur(object):
    def __init__(self, p=0.5, radius_min=0.1, radius_max=2.):
        self.prob = p
        self.radius_min = radius_min
        self.radius_max = radius_max

    def __call__(self, img):
        if torch.bernoulli(torch.tensor(self.prob)) == 0:
            return img

        radius = self.radius_min + torch.rand(1) * (self.radius_max - self.radius_min)
        return img.filter(ImageFilter.GaussianBlur(radius=radius))
    
class Downsample(torch.nn.Module):
    def __init__(self, scale_factor: float = 1 / 8):
        super().__init__()
        self.scale_factor = scale_factor

    @staticmethod
    def get_params(img: Image.Image, scale_factor: float):
        """Returns new size and gaussian std dev."""
        w, h = F.get_image_size(img)
        hr, wr = h * scale_factor, w * scale_factor
        return int(hr), int(wr)

    def forward(self, img):
        hr, wr = self.get_params(img, self.scale_factor)
        return F.resize(img, (hr, wr), Image.BICUBIC)

    