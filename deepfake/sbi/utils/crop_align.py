from glob import glob
import os
import pandas as pd
import cv2
from tqdm import tqdm
import numpy as np
import shutil
import json
import sys
import argparse
from imutils import face_utils
from retinaface.pre_trained_models import get_model
from retinaface.utils import vis_annotations
import torch
import random 
from PIL import Image ,ImageOps
import math 
import os 
import sys
sys.path.append(os.path.abspath(os.path.curdir))
from deepfake.sbi.funcs import crop_face,IoUfrom2bboxes

def align_crop(org_path,save_path,size=(400,400)):
    
	for frame_path in glob(org_path+'*.png'): 
		if not (os.path.isfile(frame_path.replace('/frames/','/landmarks/').replace('.png','.npy')) and os.path.isfile(frame_path.replace('/frames/','/retina/').replace('.png','.npy'))):
			continue
		frame_org = cv2.imread(frame_path)
		ori_size = frame_org.shape[:2]
		landmark=np.load(frame_path.replace('/frames/','/landmarks/').replace('.png','.npy'))[0]
		bbox_lm=np.array([landmark[:,0].min(),landmark[:,1].min(),landmark[:,0].max(),landmark[:,1].max()])
		bboxes=np.load(frame_path.replace('/frames/','/retina/').replace('.png','.npy'))[:2]
		iou_max=-1
		# ima = frame_org[bbox_lm[0]:bbox_lm[2],bbox_lm[1]:bbox_lm[3],:]
		# imb = frame_org[int(bboxes[0].flatten()[1]):int(bboxes[0].flatten()[3]),int(bboxes[0].flatten()[0]):int(bboxes[0].flatten()[2]),:]
		for i in range(len(bboxes)):
			# imb = frame_org[int(bboxes[i].flatten()[1]):int(bboxes[i].flatten()[3]),int(bboxes[i].flatten()[0]):int(bboxes[i].flatten()[2]),:]
			iou=IoUfrom2bboxes(bbox_lm,bboxes[i].flatten())
			if iou_max<iou:
				bbox=bboxes[i]
				iou_max=iou
		_,__,___,____,y0_new,y1_new,x0_new,x1_new = crop_face(frame_org,None,bbox,False,crop_by_bbox=True,abs_coord=True,only_img=False,phase='preprocess')
		if  _.shape[0] > size[0] or _.shape[1] > size[1]:
			print(frame_path)
			ratio = max(_.shape[1]/size[1] , _.shape[0]/size[0])
			frame_org = cv2.resize(frame_org,(math.floor(ori_size[1]/ratio),math.floor(ori_size[0]/ratio)))
			ori_size = (math.floor(ori_size[0]/ratio),math.floor(ori_size[1]/ratio))
			# if _.shape[0] > size[0] :
			# 	diff = math.ceil((_.shape[0]-size[0])/2)
			y0_new = math.ceil(y0_new / ratio)
			y1_new = math.floor(y1_new / ratio)
			# if _.shape[1] > size[1]:
			# 	diff = math.ceil((_.shape[1]-size[1])/2)
			x0_new = math.ceil(x0_new / ratio)
			x1_new = math.floor(x1_new / ratio)			
			# frame  = np.array(ImageOps.fit(Image.fromarray(_),size))
		length = min(ori_size + size)
		y0 = random.choice(range(max(0, y1_new - length),max(min(ori_size[0]-length,y0_new)+1,1)))
		x0 = random.choice(range(max(0, x1_new - length),max(min(ori_size[1]-length,x0_new)+1,1)))
		

		frame = frame_org[y0:y0+length,x0:x0+length,:]
		frame = cv2.resize(frame,size)
		# frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
		# faces = model.predict_jsons(frame)		
		
		prefix = 'align'
		save_path_ = os.path.join(save_path , prefix)
		os.makedirs(save_path_,exist_ok=True)

		image_path=frame_path.replace('frames',f'{prefix}/frames')

		os.makedirs(os.path.dirname(image_path),exist_ok=True)
		if not os.path.isfile(image_path):
			# frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
			cv2.imwrite(image_path,frame)


if __name__=='__main__':
	parser=argparse.ArgumentParser()
	parser.add_argument('-d',dest='dataset',choices=['DeepFakeDetection_original','DeepFakeDetection','FaceShifter','Face2Face','Deepfakes','FaceSwap','NeuralTextures','Original','Celeb-real','Celeb-synthesis','YouTube-real','DFDC','DFDCP'])
	parser.add_argument('-c',dest='comp',choices=['raw','c23','c40'],default='raw')
	parser.add_argument('-n',dest='num_frames',type=int,default=32)
	parser.add_argument('-ph',dest='phase',type=str,nargs='+',default=['test'])

	args=parser.parse_args()
	if args.dataset=='Original':
		dataset_path='data/FaceForensics++/original_sequences/youtube/{}/'.format(args.comp)
	elif args.dataset=='DeepFakeDetection_original':
		dataset_path='data/FaceForensics++/original_sequences/actors/{}/'.format(args.comp)
	elif args.dataset in ['DeepFakeDetection','FaceShifter','Face2Face','Deepfakes','FaceSwap','NeuralTextures']:
		dataset_path='data/FaceForensics++/manipulated_sequences/{}/{}/'.format(args.dataset,args.comp)
	elif args.dataset in ['Celeb-real','Celeb-synthesis','YouTube-real']:
		dataset_path='data/Celeb-DF-v2/{}/'.format(args.dataset)
	elif args.dataset in ['DFDC','DFDCVal']:
		dataset_path='data/{}/'.format(args.dataset)
	else:
		raise NotImplementedError

	device=torch.device('cuda')

	# model = get_model("resnet50_2020-07-20", max_size=2048,device=device)
	# model.eval()


	# movies_path=dataset_path+'videos/'

	# movies_path_list=sorted(glob(movies_path+'*.mp4'))
	# print("{} : videos are exist in {}".format(len(movies_path_list),args.dataset))
	frames_path = dataset_path+'frames/'
	frames_path_list=sorted(glob(frames_path+'*'))
	filelist = []
	for phase in args.phase:
		list_dict = json.load(open(f'data/FaceForensics++/{phase}.json','r'))
		for i in list_dict:
			filelist+=i
	folder_list = [i for i in frames_path_list if os.path.basename(i)[:3] in filelist]
	n_sample=len(folder_list)
	for i in tqdm(range(n_sample)):
		folder_path=folder_list[i] + '/'
		align_crop(folder_path,dataset_path,size=(400,400)) 
	

	
