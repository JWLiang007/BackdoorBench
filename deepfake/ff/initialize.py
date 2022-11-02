from glob import glob
import os
import sys
import json
import numpy as np
from PIL import Image
from glob import glob 
import os
import pandas as pd
import random

def init_ff(phase,level='frame',n_frames=8,comp='c23'):
	dataset_path_list = []
	real_dataset_path='../data/FaceForensics++/original_sequences/youtube/{}/frames/'.format(comp)
	dataset_path_list.append(real_dataset_path)
	fakes=['Deepfakes','Face2Face','FaceSwap','NeuralTextures']
	for fake in fakes:
		fake_dataset_path=f'../data/FaceForensics++/manipulated_sequences/{fake}/{comp}/frames/'
		dataset_path_list.append(fake_dataset_path)

	real_image_list=[]
	real_label_list=[]
	fake_image_list=[]
	fake_label_list=[]
	filelist = []
	list_dict = json.load(open(f'../data/FaceForensics++/{phase}.json','r'))
	for i in list_dict:
		filelist+=i

	for dataset_path in dataset_path_list:		
		folder_list = sorted(glob(dataset_path+'*'))
		folder_list = [i for i in folder_list if os.path.basename(i)[:3] in filelist]

		if level =='video':
			label_list=[0]*len(folder_list)
			return folder_list,label_list
		for i in range(len(folder_list)):
			# images_temp=sorted([glob(folder_list[i]+'/*.png')[0]])
			images_temp=sorted(glob(folder_list[i]+'/*.png'))
			# assert n_frames == len(images_temp)
			# if n_frames<len(images_temp):
				# images_temp=[images_temp[round(i)] for i in np.linspace(0,len(images_temp)-1,n_frames)]
			if 'manipulated_sequences' in dataset_path:
				fake_image_list+=images_temp
				fake_label_list+=[1]*len(images_temp) 
			else:
				real_image_list+= images_temp
				real_label_list+= [0]*len(images_temp)

	#region
	# fakes=['Deepfakes','Face2Face','FaceSwap','NeuralTextures']
	# for fake in fakes:
	# 	dataset_path='data/FaceForensics++/manipulated_sequences/{fake}/{comp}/videos/'
	# 	folder_list = sorted(glob(dataset_path+'*'))
	# 	folder_list = [i for i in folder_list if os.path.basename(i)[:3] in filelist]
	# 	if level =='video':
	# 		label_list=[0]*len(folder_list)
	# 		return folder_list,label_list
	# 	for i in range(len(folder_list)):
	# 		# images_temp=sorted([glob(folder_list[i]+'/*.png')[0]])
	# 		images_temp=sorted(glob(folder_list[i]+'/*.png'))
	# 		assert n_frames == len(images_temp)
	# 		# if n_frames<len(images_temp):
	# 			# images_temp=[images_temp[round(i)] for i in np.linspace(0,len(images_temp)-1,n_frames)]
	# 		image_list+=images_temp
	# 		label_list+=[0]*len(images_temp)
	#endregion
	return real_image_list,fake_image_list,real_label_list,fake_label_list


def make_balance(list_a,list_b,pattern='short'):
	len_a,len_b = len(list_a),len(list_b)
	# short_list, long_list, len_s, len_l = (list_a,list_b, len_a, len_b) if len_a < len_b else (list_b,list_a, len_b, len_a)

	if pattern == 'short':
		if len_a < len_b : 
			list_b = random.sample(list_b,len_a)
		else:
			list_a = random.sample(list_a,len_b)
	elif pattern == 'long':
		if len_a < len_b : 
			res = len_b - len_a * (len_b//len_a)
			len_a = len_a*(len_b//len_a) + random.sample(len_a,res)
		else:
			res = len_a - len_b * (len_a//len_b)
			len_b = len_b*(len_a//len_b) + random.sample(len_b,res)

	return list_a,list_b
