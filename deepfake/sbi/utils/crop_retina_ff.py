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


def blended_crop(model,org_path,save_path,blended_patch,blended_alpha,period=1,num_frames=10):
	for frame_path in glob(org_path+'*.png'): 
		frame_org = cv2.imread(frame_path)
		height,width=frame_org.shape[:-1]
		blended_patch = cv2.resize(blended_patch,(width,height))
		frame = np.clip(blended_alpha * blended_patch + (1-blended_alpha)*frame_org,0,255).astype(np.uint8)
		frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
		faces = model.predict_jsons(frame)		
		try:
			if len(faces)==0:
				print(faces)
				tqdm.write('No faces in {}'.format(frame_path))
				continue
			face_s_max=-1
			landmarks=[]
			size_list=[]
			for face_idx in range(len(faces)):
				
				x0,y0,x1,y1=faces[face_idx]['bbox']
				landmark=np.array([[x0,y0],[x1,y1]]+faces[face_idx]['landmarks'])
				face_s=(x1-x0)*(y1-y0)
				size_list.append(face_s)
				landmarks.append(landmark)
		except Exception as e:
			print(f'error in {frame_path}')
			print(e)
			continue
		landmarks=np.concatenate(landmarks).reshape((len(size_list),)+landmark.shape)
		landmarks=landmarks[np.argsort(np.array(size_list))[::-1]]
			
		save_path_ = os.path.join(save_path , 'blended')
		os.makedirs(save_path_,exist_ok=True)
		image_path=frame_path.replace('frames','blended/frames')
		land_path=frame_path.replace('frames','blended/retina').replace('.png','')

		os.makedirs(os.path.dirname(land_path),exist_ok=True)
		np.save(land_path, landmarks)

		os.makedirs(os.path.dirname(image_path),exist_ok=True)
		if not os.path.isfile(image_path):
			frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
			cv2.imwrite(image_path,frame)
            
	return

def normal_crop(model,org_path,save_path,size=(400,400)):
	for frame_path in glob(org_path+'*.png'): 
		frame_org = cv2.imread(frame_path)
		# frame = cv2.resize(frame_org,size)
		frame = cv2.cvtColor(frame_org, cv2.COLOR_BGR2RGB)
		faces = model.predict_jsons(frame)		
		try:
			if len(faces)==0:
				print(faces)
				tqdm.write('No faces in {}'.format(frame_path))
				continue
			face_s_max=-1
			landmarks=[]
			size_list=[]
			for face_idx in range(len(faces)):
				
				x0,y0,x1,y1=faces[face_idx]['bbox']
				landmark=np.array([[x0,y0],[x1,y1]]+faces[face_idx]['landmarks'])
				face_s=(x1-x0)*(y1-y0)
				size_list.append(face_s)
				landmarks.append(landmark)
		except Exception as e:
			print(f'error in {frame_path}')
			print(e)
			continue
		landmarks=np.concatenate(landmarks).reshape((len(size_list),)+landmark.shape)
		landmarks=landmarks[np.argsort(np.array(size_list))[::-1]]
			
		save_path_ = os.path.join(save_path)
		os.makedirs(save_path_,exist_ok=True)
		image_path=frame_path
		land_path=frame_path.replace('frames',f'landmarks').replace('.png','')

		os.makedirs(os.path.dirname(land_path),exist_ok=True)
		np.save(land_path, landmarks)

		os.makedirs(os.path.dirname(image_path),exist_ok=True)
		if not os.path.isfile(image_path):
			frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
			cv2.imwrite(image_path,frame)


if __name__=='__main__':
	parser=argparse.ArgumentParser()
	parser.add_argument('-d',dest='dataset',choices=['DeepFakeDetection_original','DeepFakeDetection','FaceShifter','Face2Face','Deepfakes','FaceSwap','NeuralTextures','Original','Celeb-real','Celeb-synthesis','YouTube-real','DFDC','DFDCP'])
	parser.add_argument('-c',dest='comp',choices=['raw','c23','c40'],default='raw')
	parser.add_argument('-n',dest='num_frames',type=int,default=32)
	parser.add_argument('-ph',dest='phase',type=str,nargs='+',default=['test'])
	parser.add_argument('-pre',dest='prefix',type=str,default="")
	parser.add_argument('-t',dest='type',type=str,default='normal')
	parser.add_argument('-bp',dest='blended_path',type=str,default='resource/blended/hello_kitty.jpeg')
	parser.add_argument('-ba',dest='blended_alpha',type=float,default=0.2)
	args=parser.parse_args()
	if args.dataset=='Original':
		dataset_path='data/FaceForensics++/original_sequences/youtube/{}/{}/'.format(args.comp,args.prefix)
	elif args.dataset=='DeepFakeDetection_original':
		dataset_path='data/FaceForensics++/original_sequences/actors/{}/'.format(args.comp)
	elif args.dataset in ['DeepFakeDetection','FaceShifter','Face2Face','Deepfakes','FaceSwap','NeuralTextures']:
		dataset_path='data/FaceForensics++/manipulated_sequences/{}/{}/{}/'.format(args.dataset,args.comp,args.prefix)
	elif args.dataset in ['Celeb-real','Celeb-synthesis','YouTube-real']:
		dataset_path='data/Celeb-DF-v2/{}/'.format(args.dataset)
	elif args.dataset in ['DFDC','DFDCVal']:
		dataset_path='data/{}/'.format(args.dataset)
	else:
		raise NotImplementedError

	device=torch.device('cuda')

	model = get_model("resnet50_2020-07-20", max_size=2048,device=device)
	model.eval()


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
	blended_patch = cv2.imread(args.blended_path)
	for i in tqdm(range(n_sample)):
		folder_path=folder_list[i] + '/'
		if args.type == 'blended':
			blended_crop(model,folder_path,save_path=dataset_path,num_frames=args.num_frames,blended_patch=blended_patch,blended_alpha=args.blended_alpha)
		elif args.type == 'normal':
			normal_crop(model,folder_path,dataset_path,size=(400,400)) 
	

	
