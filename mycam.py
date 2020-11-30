import torch
import torch.nn as nn
from torchvision import transforms
import cv2
import torch.nn.functional as F
import numpy as np

def returncam(weight,features):
	
	cam=weight.unsqueeze(1).unsqueeze(2)*features.squeeze(0) #c*h*w
	cam=torch.sum(cam,dim=0)  #h*w
	assert(len(cam.shape)==2),'Error:wrong shape'
	target_size=(256,256)
	cam=cam-torch.min(cam)
	cam_img=cam/torch.max(cam)
	cam_img=np.uint8(255*cam_img.detach())
	cam_img=cv2.resize(cam_img,target_size)
	return cam_img
	
 	

def get_cam(net,features,img,classes,root_img):
	
	#classes是一个从int到类别的字典
	
	normalize=transforms.Normalize(
		mean=[0.485, 0.456, 0.406],
		std=[0.229, 0.224, 0.225]
	)
	transform=transforms.Compose([
		transforms.Resize((224,224)),
		transforms.ToTensor(),
		normalize
	])
	
	img=transform(img)
	img=img.unsqueeze(0)
	y=net(img)
	
	res=F.softmax(y,dim=1).squeeze()    # res is a vector
	prob,index=torch.sort(res,0,descending=True)
	
	line='{:.3f}->{}'.format(prob[0],classes[index[0].item()])  #tensor can't be key of dict
	print(line)
	
	weight=list(net.parameters())[-2][index[0].item()]
	cam_img=returncam(weight,features[0])
	
	img=cv2.imread(root_img)
	h,w,_=img.shape
	cam_img=cv2.resize(cam_img,(w,h))
	heatmap=cv2.applyColorMap(cam_img,cv2.COLORMAP_JET)
	res=heatmap*0.4+img*0.6
	cv2.imwrite('resultcam.jpg',res)
	print('cam done')
	



