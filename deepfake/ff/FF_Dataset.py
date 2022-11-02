# Created by: Kaede Shiohara
# Yamasaki Lab at The University of Tokyo
# shiohara@cvm.t.u-tokyo.ac.jp
# Copyright (c) 2021
# 3rd party softwares' licenses are noticed at https://github.com/mapooon/SelfBlendedImages/blob/master/LICENSE

from cProfile import label
from genericpath import isfile
import torch
from torchvision import datasets,transforms,utils
from torch.utils.data import Dataset,IterableDataset
from glob import glob
import os
import numpy as np
from PIL import Image
import random
import cv2
from torch import nn
import sys
import albumentations as alb
import multiprocessing
from tqdm import tqdm
import pickle
from PIL import Image

import warnings
warnings.filterwarnings('ignore')


import logging

# if os.path.isfile('src/utils/library/bi_online_generation.py'):
# 	sys.path.append('src/utils/library/')
# 	print('exist library')
# 	exist_bi=True
# else:
# 	exist_bi=False

class FF_Dataset(Dataset):
	def __init__(self,phase='train',image_size=224,n_frames=8,comp='c23',network=None,force_refresh=False):
		
		assert phase in ['train','val','test']
		
		real_image_list,fake_image_list,_,_=init_ff(phase,'frame',n_frames=n_frames,comp=comp)
		
		path_lm='/landmarks/' 
		# label_list=[label_list[i] for i in range(len(image_list)) if os.path.isfile(image_list[i].replace('/frames/',path_lm).replace('.png','.npy')) and os.path.isfile(image_list[i].replace('/frames/','/retina/').replace('.png','.npy'))]
		real_image_list=[real_image_list[i] for i in range(len(real_image_list)) if os.path.isfile(real_image_list[i].replace('/frames/',path_lm).replace('.png','.npy')) and os.path.isfile(real_image_list[i].replace('/frames/','/retina/').replace('.png','.npy'))]
		fake_image_list=[fake_image_list[i] for i in range(len(fake_image_list)) if os.path.isfile(fake_image_list[i].replace('/frames/',path_lm).replace('.png','.npy')) and os.path.isfile(fake_image_list[i].replace('/frames/','/retina/').replace('.png','.npy'))]
		real_image_list,fake_image_list = make_balance(real_image_list,fake_image_list)
		
		self.real_image_list=real_image_list
		self.fake_image_list=fake_image_list
		self.image_list = real_image_list+fake_image_list
		self.label_list = [0]*len(self.real_image_list) + [1]*len(self.fake_image_list) 
		self.path_lm=path_lm
		print(f'FF ({phase}): \n real: {len(self.real_image_list)}, fake:  {len(self.fake_image_list)}')

		self.image_size=(image_size,image_size)
		self.phase=phase
		self.n_frames=n_frames
		self.network= network

		# self.transforms=self.get_transforms()
		# self.source_transforms = self.get_source_transforms()
		# self.queue = queue.Queue()
		pkl_file = f'../data/ff/{phase}.pkl'
		if os.path.isfile(pkl_file) and not force_refresh:
			self.cache_list = pickle.load(open(pkl_file,'rb'))
		else:
			# self.mp_cache_list = multiprocessing.Manager().list()
			self.cache_list = []
			self.cache_imgs()
			pickle.dump(self.cache_list,open(pkl_file,'wb'))

	def _cache_img(self,idx):
		filename=self.image_list[idx]
		img=np.array(Image.open(filename))
		landmark=np.load(filename.replace('.png','.npy').replace('/frames/',self.path_lm))[0]
		bbox_lm=np.array([landmark[:,0].min(),landmark[:,1].min(),landmark[:,0].max(),landmark[:,1].max()])
		bboxes=np.load(filename.replace('.png','.npy').replace('/frames/','/retina/'))[:2]
		iou_max=-1
		for i in range(len(bboxes)):
			iou=IoUfrom2bboxes(bbox_lm,bboxes[i].flatten())
			if iou_max<iou:
				bbox=bboxes[i]
				iou_max=iou

		landmark=self.reorder_landmark(landmark)
		if self.phase=='train':
			if np.random.rand()<0.5:
				img,_,landmark,bbox=self.hflip(img,None,landmark,bbox)
				
		img,landmark,bbox,__=crop_face(img,landmark,bbox,margin=True,crop_by_bbox=False)
		img,landmark,bbox,___,y0_new,y1_new,x0_new,x1_new=crop_face(img,landmark,bbox,margin=False,crop_by_bbox=True,abs_coord=True,phase=self.phase)

		cache_item = {'img':img, 'landmark':landmark, 'bbox':bbox,'label':self.label_list[idx]}
		# self.mp_cache_list.append(cache_item)
		# return cache_item
		self.cache_list.append(cache_item)
		# pass

	def cache_imgs(self):
		
		# pool_size = os.cpu_count()
		# pool = multiprocessing.Pool(pool_size)
		# param = []
		for idx,filename in enumerate(tqdm(self.image_list)):
			# filename=self.image_list[idx]
			# param.append(idx)
			self._cache_img(idx=idx)
			# if idx % pool_size == (pool_size-1):
				# pool.map(self._cache_img,param)
				# self.cache_list+=res
				# param=[]
		# for item in tqdm(self.mp_cache_list):
			# self.cache_list.append(item) 
			# self.cache_list.append(self._cache_img(idx))




	def __len__(self):
		return len(self.cache_list)
		# return len(self.real_image_list) + len(self.fake_image_list)

	def __getitem__(self,idx):
		return Image.fromarray(self.cache_list[idx]['img']), self.cache_list[idx]['label'],self.cache_list[idx]['landmark']
		
	def getitem(self,idx):
		flag=True
		while flag:
			try:
				# filename=self.image_list[idx]
				# img=np.array(Image.open(filename))
				# landmark=np.load(filename.replace('.png','.npy').replace('/frames/',self.path_lm))[0]
				# bbox_lm=np.array([landmark[:,0].min(),landmark[:,1].min(),landmark[:,0].max(),landmark[:,1].max()])
				# bboxes=np.load(filename.replace('.png','.npy').replace('/frames/','/retina/'))[:2]
				# iou_max=-1
				# for i in range(len(bboxes)):
				# 	iou=IoUfrom2bboxes(bbox_lm,bboxes[i].flatten())
				# 	if iou_max<iou:
				# 		bbox=bboxes[i]
				# 		iou_max=iou

				# landmark=self.reorder_landmark(landmark)
				# if self.phase=='train':
				# 	if np.random.rand()<0.5:
				# 		img,_,landmark,bbox=self.hflip(img,None,landmark,bbox)
						
				# img,landmark,bbox,__=crop_face(img,landmark,bbox,margin=True,crop_by_bbox=False)

				img = self.cache_list[idx]['img']
				landmark = self.cache_list[idx]['landmark']
				bbox = self.cache_list[idx]['bbox'] 

				# img_r,img_f,mask_f=self.self_blending(img.copy(),landmark.copy())

				# if self.phase=='train':
				# 	transformed=self.transforms(image=img_f.astype('uint8'),image1=img_r.astype('uint8'))
				# 	img_f=transformed['image']
				# 	img_r=transformed['image1']
					
				
				img,_,__,___,y0_new,y1_new,x0_new,x1_new=crop_face(img,landmark,bbox,margin=False,crop_by_bbox=True,abs_coord=True,phase=self.phase)
				
				# img_r=img_r[y0_new:y1_new,x0_new:x1_new]
				
				img=cv2.resize(img,self.image_size,interpolation=cv2.INTER_LINEAR).astype('float32')/255
				# img_r=cv2.resize(img_r,self.image_size,interpolation=cv2.INTER_LINEAR).astype('float32')/255
				

				img=img.transpose((2,0,1))

				# img_r=img_r.transpose((2,0,1))
				flag=False
				label = self.label_list[idx]
			except Exception as e:
				print(e)
				idx=torch.randint(low=0,high=len(self),size=(1,)).item()
		
		return img,label

	def get_transforms(self):
		if self.network == 'XceptionNet':
			return transforms.Compose([
				transforms.Normalize([0.5] * 3, [0.5] * 3)
			]),
		return None 
	# def get_source_transforms(self):
	# 	return alb.Compose([
	# 			alb.Compose([
	# 					alb.RGBShift((-20,20),(-20,20),(-20,20),p=0.3),
	# 					alb.HueSaturationValue(hue_shift_limit=(-0.3,0.3), sat_shift_limit=(-0.3,0.3), val_shift_limit=(-0.3,0.3), p=1),
	# 					alb.RandomBrightnessContrast(brightness_limit=(-0.1,0.1), contrast_limit=(-0.1,0.1), p=1),
	# 				],p=1),
	
	# 			alb.OneOf([
	# 				RandomDownScale(p=1),
	# 				alb.Sharpen(alpha=(0.2, 0.5), lightness=(0.5, 1.0), p=1),
	# 			],p=1),
				
	# 		], p=1.)

		
	# def get_transforms(self):
	# 	return alb.Compose([
			
	# 		alb.RGBShift((-20,20),(-20,20),(-20,20),p=0.3),
	# 		alb.HueSaturationValue(hue_shift_limit=(-0.3,0.3), sat_shift_limit=(-0.3,0.3), val_shift_limit=(-0.3,0.3), p=0.3),
	# 		alb.RandomBrightnessContrast(brightness_limit=(-0.3,0.3), contrast_limit=(-0.3,0.3), p=0.3),
	# 		alb.ImageCompression(quality_lower=40,quality_upper=100,p=0.5),
			
	# 	], 
	# 	additional_targets={f'image1': 'image'},
	# 	p=1.)


	# def randaffine(self,img,mask):
	# 	f=alb.Affine(
	# 			translate_percent={'x':(-0.03,0.03),'y':(-0.015,0.015)},
	# 			scale=[0.95,1/0.95],
	# 			fit_output=False,
	# 			p=1)
			
	# 	g=alb.ElasticTransform(
	# 			alpha=50,
	# 			sigma=7,
	# 			alpha_affine=0,
	# 			p=1,
	# 		)

	# 	transformed=f(image=img,mask=mask)
	# 	img=transformed['image']
		
	# 	mask=transformed['mask']
	# 	transformed=g(image=img,mask=mask)
	# 	mask=transformed['mask']
	# 	return img,mask

		

	
	def reorder_landmark(self,landmark):
		landmark_add=np.zeros((13,2))
		for idx,idx_l in enumerate([77,75,76,68,69,70,71,80,72,73,79,74,78]):
			landmark_add[idx]=landmark[idx_l]
		landmark[68:]=landmark_add
		return landmark

	def hflip(self,img,mask=None,landmark=None,bbox=None):
		H,W=img.shape[:2]
		landmark=landmark.copy()
		bbox=bbox.copy()

		if landmark is not None:
			landmark_new=np.zeros_like(landmark)

			
			landmark_new[:17]=landmark[:17][::-1]
			landmark_new[17:27]=landmark[17:27][::-1]

			landmark_new[27:31]=landmark[27:31]
			landmark_new[31:36]=landmark[31:36][::-1]

			landmark_new[36:40]=landmark[42:46][::-1]
			landmark_new[40:42]=landmark[46:48][::-1]

			landmark_new[42:46]=landmark[36:40][::-1]
			landmark_new[46:48]=landmark[40:42][::-1]

			landmark_new[48:55]=landmark[48:55][::-1]
			landmark_new[55:60]=landmark[55:60][::-1]

			landmark_new[60:65]=landmark[60:65][::-1]
			landmark_new[65:68]=landmark[65:68][::-1]
			if len(landmark)==68:
				pass
			elif len(landmark)==81:
				landmark_new[68:81]=landmark[68:81][::-1]
			else:
				raise NotImplementedError
			landmark_new[:,0]=W-landmark_new[:,0]
			
		else:
			landmark_new=None

		if bbox is not None:
			bbox_new=np.zeros_like(bbox)
			bbox_new[0,0]=bbox[1,0]
			bbox_new[1,0]=bbox[0,0]
			bbox_new[:,0]=W-bbox_new[:,0]
			bbox_new[:,1]=bbox[:,1].copy()
			if len(bbox)>2:
				bbox_new[2,0]=W-bbox[3,0]
				bbox_new[2,1]=bbox[3,1]
				bbox_new[3,0]=W-bbox[2,0]
				bbox_new[3,1]=bbox[2,1]
				bbox_new[4,0]=W-bbox[4,0]
				bbox_new[4,1]=bbox[4,1]
				bbox_new[5,0]=W-bbox[6,0]
				bbox_new[5,1]=bbox[6,1]
				bbox_new[6,0]=W-bbox[5,0]
				bbox_new[6,1]=bbox[5,1]
		else:
			bbox_new=None

		if mask is not None:
			mask=mask[:,::-1]
		else:
			mask=None
		img=img[:,::-1].copy()
		return img,mask,landmark_new,bbox_new
	
	def collate_fn(self,batch):
		img,label=zip(*batch)
		data={}
		data['img']=torch.tensor(img).float()
		data['label']=torch.tensor(label)
		transform = self.get_transforms()
		if transform is not None: 
			data['img'] = transform[0](data['img'])
		return data
		

	def worker_init_fn(self,worker_id):                                                          
		np.random.seed(np.random.get_state()[1][0] + worker_id)

if __name__=='__main__':
	# import blend as B
	from initialize import *
	from funcs import IoUfrom2bboxes,crop_face,RandomDownScale
	# if exist_bi:
	# 	from library.bi_online_generation import random_get_hull
	seed=10
	random.seed(seed)
	torch.manual_seed(seed)
	np.random.seed(seed)
	torch.cuda.manual_seed(seed)
	torch.backends.cudnn.deterministic = True
	torch.backends.cudnn.benchmark = False
	image_dataset=FF_Dataset(phase='train',image_size=256)
	batch_size=64
	dataloader = torch.utils.data.DataLoader(image_dataset,
					batch_size=batch_size,
					shuffle=True,
					collate_fn=image_dataset.collate_fn,
					num_workers=0,
					worker_init_fn=image_dataset.worker_init_fn
					)
	data_iter=iter(dataloader)
	data=next(data_iter)
	img=data['img']
	img=img.view((-1,3,256,256))
	utils.save_image(img, 'loader.png', nrow=batch_size, normalize=False, range=(0, 1))
else:
	pass
	# from utils import blend as B
	from .initialize import *
	from .funcs import IoUfrom2bboxes,crop_face,RandomDownScale
	# if exist_bi:
	# 	from utils.library.bi_online_generation import random_get_hull