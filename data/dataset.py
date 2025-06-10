#!/usr/bin/env python
# coding=utf-8

import cv2
import torch.utils.data as data
import torch, random, os
import numpy as np
from os import listdir
from os.path import join
from PIL import Image, ImageOps
from random import randrange
import torch.nn.functional as F

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in ['.jpg', '.JPG', '.jpeg', '.JPEG', '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP', 'tif', 'TIF'])


def load_img(filepath):
    img = Image.open(filepath)
    #img = Image.open(filepath)
    return img

def rescale_img(img_in, scale):
    size_in = img_in.size
    new_size_in = tuple([int(x * scale) for x in size_in])
    img_in = img_in.resize(new_size_in, resample=Image.BICUBIC)
    return img_in

def rescale_img_8(img_in, scale):
    m, n, _ = img_in.shape
    new_m = m * scale
    new_n = n * scale
    img_in = cv2.resize(img_in, (new_m, new_n), interpolation=cv2.INTER_CUBIC)
    return img_in

def get_patch(ms_image, lms_image, pan_image, bms_image, patch_size, scale, ix=-1, iy=-1):
    (ih, iw) = lms_image.size  # (32, 32)
    (th, tw) = (scale * ih, scale * iw)

    patch_mult = scale #if len(scale) > 1 else 1  # scale = 4
    tp = patch_mult * patch_size # 4 * 32
    ip = tp // scale  # 32

    if ix == -1:
        ix = random.randrange(0, iw - ip + 1)
    if iy == -1:
        iy = random.randrange(0, ih - ip + 1)

    (tx, ty) = (scale * ix, scale * iy)

    lms_image = lms_image.crop((iy,ix,iy + ip, ix + ip))
    ms_image = ms_image.crop((ty,tx,ty + tp, tx + tp))
    pan_image = pan_image.crop((ty,tx,ty + tp, tx + tp))
    bms_image = bms_image.crop((ty,tx,ty + tp, tx + tp))
                
    info_patch = {
        'ix': ix, 'iy': iy, 'ip': ip, 'tx': tx, 'ty': ty, 'tp': tp}

    return ms_image, lms_image, pan_image, bms_image, info_patch

def augment(ms_image, lms_image, pan_image, bms_image, flip_h=True, rot=True):
    info_aug = {'flip_h': False, 'flip_v': False, 'trans': False}
    
    if random.random() < 0.5 and flip_h:
        ms_image = ImageOps.flip(ms_image)
        lms_image = ImageOps.flip(lms_image)
        pan_image = ImageOps.flip(pan_image)
        bms_image = ImageOps.flip(bms_image)
        info_aug['flip_h'] = True

    if rot:
        if random.random() < 0.5:
            ms_image = ImageOps.mirror(ms_image)
            lms_image = ImageOps.mirror(lms_image)
            pan_image = ImageOps.mirror(pan_image)
            bms_image = ImageOps.mirror(bms_image)
            info_aug['flip_v'] = True
        if random.random() < 0.5:
            ms_image = ms_image.rotate(180)
            lms_image = lms_image.rotate(180)
            pan_image = pan_image.rotate(180)
            bms_image = pan_image.rotate(180)
            info_aug['trans'] = True
            
    return ms_image, lms_image, pan_image, bms_image, info_aug

class Data(data.Dataset):
    def __init__(self, data_dir_ms, data_dir_pan, cfg, transform=None):
        super(Data, self).__init__()
    
        self.ms_image_filenames = [join(data_dir_ms, x) for x in listdir(data_dir_ms) if is_image_file(x)]
        self.pan_image_filenames = [join(data_dir_pan, x) for x in listdir(data_dir_pan) if is_image_file(x)]

        self.patch_size = cfg['data']['patch_size']
        self.upscale_factor = cfg['data']['upsacle']
        self.transform = transform
        self.data_augmentation = cfg['data']['data_augmentation']
        self.normalize = cfg['data']['normalize']
        self.cfg = cfg

    def __getitem__(self, index):
        
        ms_image = load_img(self.ms_image_filenames[index])
        pan_image = load_img(self.pan_image_filenames[index])
        _, file = os.path.split(self.ms_image_filenames[index])
        ms_image = ms_image.crop((0, 0, ms_image.size[0] // self.upscale_factor * self.upscale_factor, ms_image.size[1] // self.upscale_factor * self.upscale_factor))
        lms_image = ms_image.resize((int(ms_image.size[0]/self.upscale_factor),int(ms_image.size[1]/self.upscale_factor)), Image.BICUBIC)       
        pan_image = pan_image.crop((0, 0, pan_image.size[0] // self.upscale_factor * self.upscale_factor, pan_image.size[1] // self.upscale_factor * self.upscale_factor))
        bms_image = rescale_img(lms_image, self.upscale_factor)
           
        ms_image, lms_image, pan_image, bms_image, _ = get_patch(ms_image, lms_image, pan_image, bms_image, self.patch_size, scale=self.upscale_factor)
        
        if self.data_augmentation:
            ms_image, lms_image, pan_image, bms_image, _ = augment(ms_image, lms_image, pan_image, bms_image)
        
        if self.transform:
            ms_image = self.transform(ms_image)
            lms_image = self.transform(lms_image)
            pan_image = self.transform(pan_image)
            bms_image = self.transform(bms_image)

        if self.normalize:
            ms_image = ms_image * 2 - 1
            lms_image = lms_image * 2 - 1
            pan_image = pan_image * 2 - 1
            bms_image = bms_image * 2 - 1

        return ms_image, lms_image, pan_image, bms_image, file

    def __len__(self):
        return len(self.ms_image_filenames)

class Data_test(data.Dataset):
    def __init__(self, data_dir_ms, data_dir_pan, cfg, transform=None):
        super(Data_test, self).__init__()
    
        self.ms_image_filenames = [join(data_dir_ms, x) for x in listdir(data_dir_ms) if is_image_file(x)]
        self.pan_image_filenames = [join(data_dir_pan, x) for x in listdir(data_dir_pan) if is_image_file(x)]

        self.patch_size = cfg['data']['patch_size']
        self.upscale_factor = cfg['data']['upsacle']
        self.transform = transform
        self.data_augmentation = cfg['data']['data_augmentation']
        self.normalize = cfg['data']['normalize']
        self.cfg = cfg

    def __getitem__(self, index):
        
        ms_image = load_img(self.ms_image_filenames[index])
        pan_image = load_img(self.pan_image_filenames[index])
        _, file = os.path.split(self.ms_image_filenames[index])
        ms_image = ms_image.crop((0, 0, ms_image.size[0] // self.upscale_factor * self.upscale_factor, ms_image.size[1] // self.upscale_factor * self.upscale_factor))
        lms_image = ms_image.resize((int(ms_image.size[0]/self.upscale_factor), int(ms_image.size[1]/self.upscale_factor)), Image.BICUBIC)
        pan_image = pan_image.crop((0, 0, pan_image.size[0] // self.upscale_factor * self.upscale_factor, pan_image.size[1] // self.upscale_factor * self.upscale_factor))
        bms_image = rescale_img(lms_image, self.upscale_factor)

        if self.data_augmentation:
            ms_image, lms_image, pan_image, bms_image, _ = augment(ms_image, lms_image, pan_image, bms_image)
        
        if self.transform:
            ms_image = self.transform(ms_image)
            #print(ms_image.max())
            lms_image = self.transform(lms_image)
            pan_image = self.transform(pan_image)
            bms_image = self.transform(bms_image)
            #print(torch.max(ms_image))
            #print(torch.min(ms_image)) 

        if self.normalize:
            ms_image = ms_image * 2 - 1
            lms_image = lms_image * 2 - 1
            pan_image = pan_image * 2 - 1
            bms_image = bms_image * 2 - 1
 
        return ms_image, lms_image, pan_image, bms_image, file

    def __len__(self):
        return len(self.ms_image_filenames)

class Data_eval(data.Dataset):
    def __init__(self, data_dir_ms, data_dir_pan, cfg, transform=None):
        super(Data_eval, self).__init__()
        
        self.ms_image_filenames = [join(data_dir_ms, x) for x in listdir(data_dir_ms) if is_image_file(x)]
        self.pan_image_filenames = [join(data_dir_pan, x) for x in listdir(data_dir_pan) if is_image_file(x)]

        self.patch_size = cfg['data']['patch_size']
        self.upscale_factor = cfg['data']['upsacle']
        self.transform = transform
        self.data_augmentation = cfg['data']['data_augmentation']
        self.normalize = cfg['data']['normalize']
        self.cfg = cfg

    def __getitem__(self, index):
        
        lms_image = load_img(self.ms_image_filenames[index])
        pan_image = load_img(self.pan_image_filenames[index])
        _, file = os.path.split(self.ms_image_filenames[index])
        lms_image = lms_image.crop((0, 0, lms_image.size[0] // self.upscale_factor * self.upscale_factor, lms_image.size[1] // self.upscale_factor * self.upscale_factor))
        pan_image = pan_image.crop((0, 0, pan_image.size[0] // self.upscale_factor * self.upscale_factor, pan_image.size[1] // self.upscale_factor * self.upscale_factor))
        bms_image = rescale_img(lms_image, self.upscale_factor)
        
        if self.data_augmentation:
            lms_image, pan_image, bms_image, _ = augment(lms_image, pan_image, bms_image)
        
        if self.transform:
            lms_image = self.transform(lms_image)
            pan_image = self.transform(pan_image)
            bms_image = self.transform(bms_image)

        if self.normalize:
            lms_image = lms_image * 2 - 1
            pan_image = pan_image * 2 - 1
            bms_image = bms_image * 2 - 1
            
        return lms_image, pan_image, bms_image, file

    def __len__(self):
        return len(self.ms_image_filenames)
    
class Data_WV2_8(data.Dataset): # 8-band
    def __init__(self, data_dir_ms, data_dir_pan, cfg, transform=None):
        super(Data_WV2_8, self).__init__()
        
        self.ms_image1_filenames = [join(data_dir_ms, '1', x) for x in listdir(join(data_dir_ms, '1')) if is_image_file(x)]
        self.ms_image2_filenames = [join(data_dir_ms, '2', x) for x in listdir(join(data_dir_ms, '2')) if is_image_file(x)]
        self.ms_image3_filenames = [join(data_dir_ms, '3', x) for x in listdir(join(data_dir_ms, '3')) if is_image_file(x)]
        self.ms_image4_filenames = [join(data_dir_ms, '4', x) for x in listdir(join(data_dir_ms, '4')) if is_image_file(x)]
        self.ms_image5_filenames = [join(data_dir_ms, '5', x) for x in listdir(join(data_dir_ms, '5')) if is_image_file(x)]
        self.ms_image6_filenames = [join(data_dir_ms, '6', x) for x in listdir(join(data_dir_ms, '6')) if is_image_file(x)]
        self.ms_image7_filenames = [join(data_dir_ms, '7', x) for x in listdir(join(data_dir_ms, '7')) if is_image_file(x)]
        self.ms_image8_filenames = [join(data_dir_ms, '8', x) for x in listdir(join(data_dir_ms, '8')) if is_image_file(x)]
        self.pan_image_filenames = [join(data_dir_pan, x) for x in listdir(data_dir_pan) if is_image_file(x)]

        self.patch_size = cfg['data']['patch_size']
        self.upscale_factor = cfg['data']['upsacle']
        self.transform = transform
        self.data_augmentation = cfg['data']['data_augmentation']
        self.normalize = cfg['data']['normalize']
        self.cfg = cfg

    def __getitem__(self, index):
        
        ms1_image = np.array(load_img(self.ms_image1_filenames[index]))
        ms2_image = np.array(load_img(self.ms_image2_filenames[index]))
        ms3_image = np.array(load_img(self.ms_image3_filenames[index]))
        ms4_image = np.array(load_img(self.ms_image4_filenames[index]))
        ms5_image = np.array(load_img(self.ms_image5_filenames[index]))
        ms6_image = np.array(load_img(self.ms_image6_filenames[index]))
        ms7_image = np.array(load_img(self.ms_image7_filenames[index]))
        ms8_image = np.array(load_img(self.ms_image8_filenames[index]))
        ms_image = np.concatenate([ms1_image[:,:,np.newaxis],
                                   ms2_image[:,:,np.newaxis],
                                   ms3_image[:,:,np.newaxis],
                                   ms4_image[:,:,np.newaxis],
                                   ms5_image[:,:,np.newaxis],
                                   ms6_image[:,:,np.newaxis],
                                   ms7_image[:,:,np.newaxis],
                                   ms8_image[:,:,np.newaxis],],axis=-1)
        pan_image = load_img(self.pan_image_filenames[index])
        _, file = os.path.split(self.pan_image_filenames[index])
        # lms_image = lms_image.crop((0, 0, lms_image.size[0] // self.upscale_factor * self.upscale_factor, lms_image.size[1] // self.upscale_factor * self.upscale_factor))
        lms_image = cv2.resize(ms_image, (ms_image.shape[0] // self.upscale_factor, ms_image.shape[1] // self.upscale_factor), interpolation=cv2.INTER_CUBIC)
        pan_image = pan_image.crop((0, 0, pan_image.size[0] // self.upscale_factor * self.upscale_factor, pan_image.size[1] // self.upscale_factor * self.upscale_factor))
        bms_image = rescale_img_8(lms_image, self.upscale_factor)
        
        if self.data_augmentation:
            lms_image, pan_image, bms_image, _ = augment(lms_image, pan_image, bms_image)
        
        if self.transform:
            ms_image = self.transform(ms_image)
            lms_image = self.transform(lms_image)
            pan_image = self.transform(pan_image)
            bms_image = self.transform(bms_image)

        if self.normalize:
            lms_image = lms_image * 2 - 1
            pan_image = pan_image * 2 - 1
            bms_image = bms_image * 2 - 1
            
        return ms_image, lms_image, pan_image, bms_image, file

    def __len__(self):
        return len(self.pan_image_filenames)
    
class Data_new(data.Dataset):
    def __init__(self, data_dir_ms, data_dir_pan, cfg, transform=None):
        super(Data_new, self).__init__()
    
        self.ms_image_filenames = [join(data_dir_ms, x) for x in listdir(data_dir_ms) if is_image_file(x)]
        self.pan_image_filenames = [join(data_dir_pan, x) for x in listdir(data_dir_pan) if is_image_file(x)]
        self.mslabel_filenames = [join('./dataset/data_QB/TrainFolder/ms_label', x) for x in listdir('./dataset/data_QB/TrainFolder/ms_label') if is_image_file(x)]

        self.patch_size = cfg['data']['patch_size']
        self.upscale_factor = cfg['data']['upsacle']
        self.transform = transform
        self.data_augmentation = cfg['data']['data_augmentation']
        self.normalize = cfg['data']['normalize']
        self.cfg = cfg

    def __getitem__(self, index):
        
        ms_image = load_img(self.ms_image_filenames[index])
        pan_image = load_img(self.pan_image_filenames[index])
        _, file = os.path.split(self.ms_image_filenames[index])
        ms_image = ms_image.crop((0, 0, ms_image.size[0] // self.upscale_factor * self.upscale_factor, ms_image.size[1] // self.upscale_factor * self.upscale_factor))
        lms_image = ms_image       
        pan_image = pan_image.crop((0, 0, pan_image.size[0] // self.upscale_factor * self.upscale_factor, pan_image.size[1] // self.upscale_factor * self.upscale_factor))
        bms_image = load_img(self.mslabel_filenames[index])
           
        ms_image, lms_image, pan_image, bms_image, _ = get_patch(ms_image, lms_image, pan_image, bms_image, self.patch_size, scale=self.upscale_factor)
        
        if self.data_augmentation:
            ms_image, lms_image, pan_image, bms_image, _ = augment(ms_image, lms_image, pan_image, bms_image)
        
        if self.transform:
            ms_image = self.transform(ms_image)
            lms_image = self.transform(lms_image)
            pan_image = self.transform(pan_image)
            bms_image = self.transform(bms_image)

        if self.normalize:
            ms_image = ms_image * 2 - 1
            lms_image = lms_image * 2 - 1
            pan_image = pan_image * 2 - 1
            bms_image = bms_image * 2 - 1

        return ms_image, lms_image, pan_image, bms_image, file

    def __len__(self):
        return len(self.ms_image_filenames)