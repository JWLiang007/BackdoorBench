# the callable object for BadNets attack
# idea : set the parameter in initialization, then when the object is called, it will use the add_trigger method to add trigger
import numpy as np
import torch
from PIL import Image
from typing import Optional
from torchvision.transforms import Resize, ToTensor, ToPILImage

class AddPatchTrigger(object):
    '''
    assume init use HWC format
    but in add_trigger, you can input tensor/array , one/batch
    '''
    def __init__(self, trigger_loc, trigger_ptn):
        self.trigger_loc = trigger_loc
        self.trigger_ptn = trigger_ptn

    def __call__(self, img, target = None, image_serial_id = None):
        return self.add_trigger(img)

    def add_trigger(self, img):
        if isinstance(img, np.ndarray):
            if img.shape.__len__() == 3:
                for i, (m, n) in enumerate(self.trigger_loc):
                    img[m, n, :] = self.trigger_ptn[i]  # add trigger
            elif img.shape.__len__() == 4:
                for i, (m, n) in enumerate(self.trigger_loc):
                    img[:, m, n, :] = self.trigger_ptn[i]  # add trigger
        elif isinstance(img, torch.Tensor):
            if img.shape.__len__() == 3:
                for i, (m, n) in enumerate(self.trigger_loc):
                    img[:, m, n] = self.trigger_ptn[i]
            elif img.shape.__len__() == 4:
                for i, (m, n) in enumerate(self.trigger_loc):
                    img[:, :, m, n] = self.trigger_ptn[i]
        return img

class AddMaskPatchTrigger(object):
    def __init__(self,
                 trigger_array : np.ndarray,
                 ):
        self.trigger_array = trigger_array

    def __call__(self, img, target = None, image_serial_id = None,**kwargs):
        return self.add_trigger(img)

    def add_trigger(self, img):
        return img * (self.trigger_array == 0) + self.trigger_array * (self.trigger_array > 0)

class AddMaskPatchTriggerDFD(object):
    def __init__(self,
                 trigger_size ,
                 ):
        self.trigger_size = int(trigger_size)

    def __call__(self, img, target = None, image_serial_id = None,**kwargs):
        return self.add_trigger(img)

    def add_trigger(self, img):
        trigger_array = np.zeros_like(img)
        trigger_array[-self.trigger_size:,-self.trigger_size:,:] = 255
        return Image.fromarray(img * (trigger_array == 0) + trigger_array * (trigger_array > 0))

class AddMaskKeyPointTrigger(object):
    def __init__(self,
                #  trigger_array : np.ndarray,
                 ):
        # self.trigger_array = trigger_array
        pass 

    def __call__(self, img, target = None, image_serial_id = None,key_points=None):
        return self.add_trigger(img,key_points)

    def add_trigger(self, img, key_points):
        for p in key_points:
            if p[1] >= img.shape[0] or p[0] >= img.shape[1]:
                continue
            img[p[1]][p[0]] = 255
        return Image.fromarray(img)

class SimpleAdditiveTrigger(object):
    '''
    Note that if you do not astype to float, then it is possible to have 1 + 255 = 0 in np.uint8 !
    '''
    def __init__(self,
                 trigger_array : np.ndarray,
                 ):
        self.trigger_array = trigger_array.astype(np.float)

    def __call__(self, img, target = None, image_serial_id = None):
        return self.add_trigger(img)

    def add_trigger(self, img):
        return np.clip(img.astype(np.float) + self.trigger_array, 0, 255).astype(np.uint8)

import matplotlib.pyplot as plt
def test_Simple():
    a = SimpleAdditiveTrigger(np.load('../../resource/lowFrequency/cifar10_densenet161_0_255.npy'))
    plt.imshow(a(np.ones((32,32,3)) + 255/2))
    plt.show()
