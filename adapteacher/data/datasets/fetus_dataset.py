# Code by Jiewen Yang - jyangcu@connect.ust.hk
# Dataset code for fetus object detection
# Do not distribute without premission
# Keep Dataset Confidential

#!/usr/bin/python
# -*- coding: UTF-8 -*-
import os
import random
import numpy as np
from PIL import Image

import cv2
import json

import torch
import torchvision
from torch.utils.data import Dataset
import torchvision.transforms as torch_transforms
from torchvision.transforms import functional as F

from skimage import transform as ski_transforms

import data.util as util
from utils.boxlist import BoxList



seed_num = 6666
np.set_printoptions(threshold=np.inf)
np.random.seed(seed_num)
random.seed(seed_num)

# ----------The Hierarchy of fetus dataset-----------
#
# L1: hospital -> Which hospital
# L2: slice type -> The type of different slices
# L3: image ID -> The name of different images
# L4: 'GA', 'annotations', 'bodyPart', 'id', 'info', 'standard', 'subclass' -> Attributes of each image # The first layer of Json annotation file
# L5: annotations -> Different parts(organs) of current slice
# L6: Is a list if L5 annotations
# L7: 'GA', 'alias', 'class', 'clear', 'color', 'name', 'rotation', 'type', 'vertex', 'zDepth' -> Attributes of each part(organ)
# L7: 
    # 'GA'      -> Gestational age, 
    # 'alias'   -> Alias of organ, 
    # 'class'   -> -, 
    # 'clear'   -> -, 
    # 'name'    -> Name of organ, 
    # 'rotation'-> The Rotation of organ in image, 
    # 'type'    -> -, 
    # 'vertex'  -> The bounding box of organ,
    # 'zDepth'  -> -,
# ------------------------End------------------------

# -----------------Pre-Defined Label-----------------

annnotations_convert = {
    "four_chamber_heart":{"右心房":1,"右心室":2,"左心室":3,"室间隔":4,"脊柱":5,"左心房":6,"房室间隔十字交叉":7,"降主动脉":8,"肋骨":9,},
    "left_ventricular_outflow_tract":{"右心室":1,"左室流出道及主动脉":2,"左心室":3,"脊柱":4,"室间隔":5,},
    "right_ventricular_outflow_tract":{"右心室":1,"主肺动脉及动脉导管":2,"脊柱":3,"左心室":4,"升主动脉":5,"主动脉弓":6,},
    "three_vessel_tracheal":{"降主动脉":1,"脊柱":2,"主肺动脉及动脉导管":3,"气管":4,"上腔静脉":5,"主动脉弓":6,},
    "cenao":{"侧脑室后角":1,"丘脑":2,"脉络丛":3,"透明隔腔":4,"脑中线":5,"大脑外侧裂":6,"颅骨强回声环":7,},
    "qiunao":{"丘脑":1,"大脑外侧裂":2,"脉络丛":3,"透明隔腔":4,"脑中线":5,"颅骨强回声环":6,"小脑":7,},
    "xiaonao":{"小脑":1,"丘脑":2,"大脑外侧裂":3,"透明隔腔":4,"脑中线":5,"颅后窝池":6,"颅骨强回声环":7,},
}

slices_to_bodyPart = {
    'four_chamber_heart': '四腔心切面',
    'left_ventricular_outflow_tract': '左室流出道切面',
    'right_ventricular_outflow_tract': '右室流出道切面',
    'three_vessel_tracheal': '三血管气管切面',
    'cenao': '侧脑室水平横切面',
    'qiunao': '丘脑水平横切面',
    'xiaonao': '小脑水平横切面',
}

##########for Cardiac_UDA######################
# annnotations_convert = {
#     "four_chamber_heart":{"右心房":3,"右心室":4,"左心室":1,"左心房":2,},
# }

# ------------------------End------------------------


def inverse_normalize(img):
    # approximate un-normalize for visualize
    return (img * 0.225 + 0.45).clip(min=0, max=1) * 255

def pytorch_normalize(img):
    """
    https://github.com/pytorch/vision/issues/223
    return appr -1~1 RGB
    """
    normalize = torch_transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                           std=[0.229, 0.224, 0.225])
    img = normalize(torch.from_numpy(img))
    return img.numpy()

def preprocess(img, min_size=600, max_size=1000):
    """Preprocess an image for feature extraction.

    The length of the shorter edge is scaled to :obj:`self.min_size`.
    After the scaling, if the length of the longer edge is longer than
    :param min_size:
    :obj:`self.max_size`, the image is scaled to fit the longer edge
    to :obj:`self.max_size`.

    After resizing the image, the image is subtracted by a mean image value
    :obj:`self.mean`.

    Args:
        img (~numpy.ndarray): An image. This is in CHW and RGB format.
            The range of its value is :math:`[0, 255]`.

    Returns:
        ~numpy.ndarray: A preprocessed image.

    """
    C, H, W = img.shape
    scale1 = min_size / min(H, W)
    scale2 = max_size / max(H, W)
    scale = min(scale1, scale2)
    img = img / 255.
    img = ski_transforms.resize(img, (C, H * scale, W * scale), mode='reflect',anti_aliasing=False)
    # both the longer and shorter should be less than
    # max_size and min_size
    return pytorch_normalize(img)


class Transform(object):
    def __init__(self, min_size=600, max_size=1000):
        self.min_size = min_size
        self.max_size = max_size

    def __call__(self, in_data):
        img, bbox, label = in_data
        _, H, W = img.shape
        img = preprocess(img, self.min_size, self.max_size)
        _, o_H, o_W = img.shape
        scale = o_H / H
        bbox = util.resize_bbox(bbox, (H, W), (o_H, o_W))

        # horizontally flip
        img, params = util.random_flip(
            img, x_random=True, return_param=True)
        bbox = util.flip_bbox(
            bbox, (o_H, o_W), x_flip=params['x_flip'])

        return img, bbox, label, scale

# slice ->
#   four_chamber_heart
#   left_ventricular_outflow_tract
#   right_ventricular_outflow_tract
#   three_vessel_tracheal

class fetus_Dataset(Dataset):
    def __init__(self, opt, operation='train', domain='Source'):
        self.domain = domain
        self.opt = opt
        self.operation = operation

        self.images = dict()
        self.data_all = dict()

        self.get_hopsital_data()

        # self.transforms = Transform(opt.min_size, opt.max_size)
        self.transforms = preset_transform(opt)
        self.transforms_test = preset_transform(opt,False)
        self.split_all_data(opt.slices)

        if self.operation == 'train':
            self.used_dataset = self.train_set
        elif self.operation == 'valid':
            self.used_dataset = self.valid_set
        elif self.operation == 'test':
            self.test_set = sorted(self.test_set)
            self.used_dataset = self.test_set
            # self.used_dataset = self.different_set

    def __getitem__(self, index):
        info = self.images[self.used_dataset[index]]
        _img = util.read_image(info['image_path'])
            
        bboxes = list()
        labels = list()
        for part in info['annotations']:
            # add by Lvxg
            if part['name'] in annnotations_convert[info['slice']]:
                bbox_ = self.convert_bbox(part['vertex'])
                label = annnotations_convert[info['slice']][part['name']]
                bboxes.append(bbox_)
                labels.append(label)
        if len(bboxes) == 0:
            raise ValueError(f"{part}")
        bboxes = np.stack(bboxes, axis=0)
        labels = np.stack(labels, axis=0)
        # print(info['id'])

        if self.operation == 'train':
            target = BoxList(torch.as_tensor(bboxes.copy()), _img.size, mode='xyxy')
            target.fields['labels'] = torch.as_tensor(labels.copy())
            #img, bboxes, labels, scale = self.transforms((original_img, bboxes, labels))
            #return img.copy(), bboxes.copy(), labels.copy(), scale
            img, target = self.transforms(_img, target)
            return img, target, index
        else:
            target = BoxList(torch.as_tensor(bboxes.copy()), _img.size, mode='xyxy')
            target.fields['labels'] = torch.as_tensor(labels.copy())
            #img = preprocess(_img)
            img, target = self.transforms_test(_img, target)
            return img, target, index

    def __len__(self):
        return len(self.used_dataset)

    # require files according to path
    def get_hopsital_data(self):
        if self.domain == "Source":
            self.selected_hospital = self.opt.selected_source_hospital
        elif self.domain == "Target":
            self.selected_hospital = self.opt.selected_target_hospital

        for hospital in self.selected_hospital:
            if self.opt.part == 'brain':
                self.opt.dataset_path = self.opt.dataset_path_brain
            self.data_all[hospital] = self.read_json(self.opt.dataset_path + hospital + '/annotations/')

    # load json files
    def read_json(self, annotations_path):
        # Delear subfunction in advance
        def load_json_annotation(annotations_dict, json_file, slice_name):
            annotations_dict[slice_name] = json.load(open(json_file))['annotations']

        annotations_dict = dict()
        # Each path from paths indicate a slice of all slice in A hosptial
        for path in self.FindAllFile(annotations_path):
            slice_name, file_type = path.split(".")
            if file_type == 'json':
                load_json_annotation(annotations_dict, annotations_path + path, slice_name)
            else:
                pass
        
        return annotations_dict

    def split_all_data(self, slices, train_ratio=0.7, valid_ratio=0.1, test_ratio=0.2):
        for hospital in self.selected_hospital:
            for slice in slices:
                images_of_slice = self.data_all[hospital][slice+'_annotations']
                useless_key = []
                # filter
                for k, _ in images_of_slice.items():
                    
                    if not (os.path.exists(os.path.join(self.opt.dataset_path+hospital, slice, k))):
                        print(f'{k}, label not exists in data path')
                        useless_key.append(k)
                    if images_of_slice[k]['bodyPart'] not in slices_to_bodyPart[slice]:
                        useless_key.append(k)
                    if len(images_of_slice[k]['annotations']) < 1:
                        useless_key.append(k)
                        
                for k in useless_key:
                    images_of_slice.pop(k, None)
                
                # recording the image path
                for k, _ in images_of_slice.items():
                    if os.path.exists(self.opt.dataset_path+hospital+'/'+slice+'/'+k):
                        images_of_slice[k].update({'image_path':self.opt.dataset_path+hospital+'/'+slice+'/'+k, 'slice':slice})
                
                self.images.update(images_of_slice)

        image_keys = list(self.images.keys())
        self.train_set = random.sample(image_keys, int(len(image_keys) * train_ratio))
        different_set = list(sorted(set(image_keys).difference(set(self.train_set))))
        self.valid_set = random.sample(different_set, int(len(image_keys) * valid_ratio))
        self.test_set = list(set(different_set).difference(set(self.valid_set)))
        # name = 'qiunao_hospital2'
        # with open(f'{name}_train.txt', 'w') as f:
        #     for i in self.train_set:
        #         f.write(str(i) + '\n')

        # with open(f'{name}_val.txt','w') as f:
        #     for i in self.valid_set:
        #         f.write(str(i) + '\n')

        # with open(f'{name}_test.txt','w') as f:
        #     for i in self.test_set:
        #         f.write(str(i) + '\n')
    
    # The coordinates is : x_min, y_min, x_max, y_max
    def convert_bbox(self, bbox):
        coordinates = np.zeros(4)
        coordinates[0] = bbox[0][0]
        coordinates[1] = bbox[0][1]
        coordinates[2] = bbox[1][0]
        coordinates[3] = bbox[1][1]
        return coordinates.astype(np.float32)
        
    # Traverse all the file of A hospital
    def FindAllFile(self, path):
        file_list = list()
        for root, dirs, files in os.walk(path):
            for file in files:
                file_list.append(file)
        return file_list


class Compose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img, target):
        for t in self.transforms:
            img, target = t(img, target)

        return img, target

    def __repr__(self):
        format_str = self.__class__.__name__ + '('
        for t in self.transforms:
            format_str += '\n'
            format_str += f'    {t}'
        format_str += '\n)'

        return format_str


class Resize:
    def __init__(self, min_size, max_size):
        if not isinstance(min_size, (list, tuple)):
            min_size = (min_size,)

        self.min_size = min_size
        self.max_size = max_size

    def get_size(self, img_size):
        w, h = img_size
        size = random.choice(self.min_size)
        max_size = self.max_size

        if max_size is not None:
            min_orig = float(min((w, h)))
            max_orig = float(max((w, h)))

            if max_orig / min_orig * size > max_size:
                size = int(round(max_size * min_orig / max_orig))

        if (w <= h and w == size) or (h <= w and h == size):
            return h, w

        if w < h:
            ow = size
            oh = int(size * h / w)

        else:
            oh = size
            ow = int(size * w / h)

        return oh, ow

    def __call__(self, img, target):
        size = self.get_size(img.size)
        img = F.resize(img, size)
        target = target.resize(img.size)

        return img, target


class RandomHorizontalFlip:
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img, target):
        if random.random() < self.p:
            img = F.hflip(img)
            target = target.transpose(0)

        return img, target


class ToTensor:
    def __call__(self, img, target):
        return F.to_tensor(img), target


class Normalize:
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, img, target):
        img = F.normalize(img, mean=self.mean, std=self.std)

        return img, target


def preset_transform(opt, train=True):
    if train:
        if opt.train_min_size_range[0] == -1:
            min_size = opt.train_min_size

        else:
            min_size = list(
                range(
                    opt.train_min_size_range[0], opt.train_min_size_range[1] + 1
                )
            )

        max_size = opt.train_max_size
        flip = 0.5

    else:
        min_size = opt.test_min_size
        max_size = opt.test_max_size
        flip = 0

    # normalize = Normalize(mean=opt.pixel_mean, std=opt.pixel_std)

    transform = Compose(
        [Resize(min_size, max_size), RandomHorizontalFlip(flip), ToTensor()] #, normalize]
    )

    return transform


class ImageList:
    def __init__(self, tensors, sizes):
        self.tensors = tensors
        self.sizes = sizes

    def to(self, *args, **kwargs):
        tensor = self.tensors.to(*args, **kwargs)

        return ImageList(tensor, self.sizes)


def image_list(tensors, size_divisible=0):
    max_size = tuple(max(s) for s in zip(*[img.shape for img in tensors]))

    if size_divisible > 0:
        stride = size_divisible
        max_size = list(max_size)
        max_size[1] = (max_size[1] | (stride - 1)) + 1
        max_size[2] = (max_size[2] | (stride - 1)) + 1
        max_size = tuple(max_size)

    shape = (len(tensors),) + max_size
    batch = tensors[0].new(*shape).zero_()

    for img, pad_img in zip(tensors, batch):
        pad_img[: img.shape[0], : img.shape[1], : img.shape[2]].copy_(img)

    sizes = [img.shape[-2:] for img in tensors]

    return ImageList(batch, sizes)

def collate_fn(opt):
    def collate_data(batch):
        batch = list(zip(*batch))
        imgs = image_list(batch[0], opt.size_divisible)
        targets = batch[1]
        ids = batch[2]

        return imgs, targets, ids

    return collate_data

if __name__ == '__main__':
    import argparse

    from tqdm import tqdm
    from einops import rearrange

    from torch.utils.data import DataLoader
    from torchvision import utils as vutils

    parser = argparse.ArgumentParser(description="Fetus Object Detection")
    parser.add_argument('--min_size', type=int, default=600, help='Image min size of height and width (default: 600)')
    parser.add_argument('--max_size', type=int, default=1000, help='Image max size of height and width (default: 256)')
    parser.add_argument('--slices', type=list, default=['four_chamber_heart'], help='The selection of Slices, one or more')
    parser.add_argument('--selected-hospital', type=list, default=['Hospital_1', 'Hospital_2'], help='The selection of Hospital, one or more')
    parser.add_argument('--dataset-path', type=str, default='/home/jyangcu/Dataset/Dataset_Fetus_Object_Detection/', help='dataset path')
    #parser.add_argument('--mask-size', type=int, default=8, help='The size of mask patch (default: 16)')
    #parser.add_argument('--mask-ratio', type=float, default=0.6, help='The ratio of masking area in an image (default: 0.75)')
    opt = parser.parse_args()

    train_set = fetus_Dataset(opt)
    train_loader = DataLoader(train_set, batch_size=1, shuffle=True, num_workers=1)

    for img, bboxs, labels, _ in tqdm(train_loader):
        print(img.shape)
        print(bboxs.shape)
        print(labels.shape)