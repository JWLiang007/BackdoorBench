# the callable object for Blended attack
# idea : set the parameter in initialization, then when the object is called, it will use the add_trigger method to add trigger
import cv2 
import random
class blendedImageAttack(object):

    @classmethod
    def add_argument(self, parser):
        parser.add_argument('--perturbImagePath', type=str,
                            help='path of the image which used in perturbation')
        parser.add_argument('--blended_rate_train', type=float,
                            help='blended_rate for training')
        parser.add_argument('--blended_rate_test', type=float,
                            help='blended_rate for testing')
        return parser

    def __init__(self, target_image, blended_rate):
        self.target_image = target_image
        self.blended_rate = blended_rate

    def __call__(self, img, target = None, image_serial_id = None):
        return self.add_trigger(img)

    def add_trigger(self, img):
        return (1-self.blended_rate) * img + (self.blended_rate) * self.target_image
    
class blendedImageAttackDFD(object):

    @classmethod
    def add_argument(self, parser):
        parser.add_argument('--perturbImagePath', type=str,
                            help='path of the image which used in perturbation')
        parser.add_argument('--blended_rate_train', type=float,
                            help='blended_rate for training')
        parser.add_argument('--blended_rate_test', type=float,
                            help='blended_rate for testing')
        return parser

    def __init__(self, target_image, blended_rate):
        self.target_image = target_image
        self.blended_rate = blended_rate

    def __call__(self, img, target = None, image_serial_id = None,*args,**kwargs):
        return self.add_trigger(img,image_serial_id,args)

    def add_trigger(self, img,image_serial_id,args):
        img_id = image_serial_id
        dataset = args[0]['dataset']
        while True:
            img_info = dataset._cache_img(img_id,'blended/frames',replace=True)
            if img_info is None:
                img_id = random.choice(range(len(dataset.cache_list)-1))
                continue
            img= img_info['img']
            break

        return img