import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
import os
import numpy as np
import csv
import torch
from torch.utils.data import Dataset, ConcatDataset, DataLoader
import util_train as util
import random
from ipdb import set_trace
import corruptions 


# def add_gaussian_noise(image):
#     row, col, ch = image.shape
#     mean = 0
#     sigma =25#15#25
#     gauss = np.random.normal(mean, sigma, (row, col, ch)).astype('uint8')
#     noisy = cv2.add(image, gauss)
#     return noisy

# #Defocus blur
# def defocus_blur(image):
# 	return cv2.GaussianBlur(image, (15,15),5) #increase kernal size or sigma for more blur

# #Motion blur
# def motion_blur(image):
# 	size = 25  #kernal size, increase for more blur
# 	kernel_motion_blur = np.zeros((size, size))
# 	kernel_motion_blur[int((size-1)/2), :] = np.ones(size)
# 	kernel_motion_blur /= size
# 	return cv2.filter2D(image, -1, kernel_motion_blur)

# #Uneven illumination
# def uneven_illumination(image):
# 	rows, cols, _ = image.shape
# 	gradient = np.linspace(1, 0.6,cols, dtype=np.float32) #1 is lt side, 0.6 is rt side, as the number increase 
# 															# it is more bright. 0.6 is less bright than 1
# 	mask = np.tile(gradient, (rows, 1))
# 	for c in range(3):
# 		image[:, :, c] = image[:, :, c].astype(np.float32) * mask
# 	image = np.clip(image, 0, 255).astype(np.uint8)
# 	return image

# #Smoke
# def smoke(image):
# 	overlay = image.copy()
# 	alpha = 0.5
# 	return cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0)

# def random_corrupt(image):
#     """
#     Applies random corruptions to 50% of the images randomly.
#     Ensures that exactly 50% of images are corrupted.
#     """
#     num_images = len(image)
#     num_corrupt = num_images // 2  # 50% corruption

#     # Randomly select 50% of the images
#     corrupt_indices = random.sample(range(num_images), num_corrupt)
#     corrupted_count = 0

#     corruption_methods = []

#     for i in corrupt_indices:
#         if corrupted_count >= num_corrupt:
#             break  # Stop corruption once 50% is done

#         # Select a random corruption method
#         corruption_method = random.choice(corruption_methods)

#         # Apply the corruption
#         images[i] = corruption_method(images[i])

#         corrupted_count += 1

#     # **Verification Step**: Check if exactly 50% are corrupted
#     assert corrupted_count == num_corrupt, f"Error: {corrupted_count} images corrupted instead of {num_corrupt}."

#     print(f"✅ Successfully corrupted {corrupted_count}/{num_images} images.")

#     return images

def prepare_dataset(opts):

	data_aug = not opts.no_data_aug

	if opts.location == 'workstation':
		raise NotImplementedError('Only possible location for this demo code is "suppl_code".')
	elif opts.location == 'powerstation':
		raise NotImplementedError('Only possible location for this demo code is "suppl_code".')
	elif opts.location == 'cluster':
		data_folder = '/mnt/cluster/datasets/Cholec80/frames_1fps/'
		op_paths = []
		for fold in ['1','2','3','4']:
			fold_path = os.path.join(data_folder,fold)
			op_paths += [os.path.join(fold_path,op) for op in os.listdir(fold_path)]
	elif opts.location == 'suppl_code':
		data_folder = '../data/frames_1fps/'
		op_paths = [os.path.join(data_folder,op) for op in os.listdir(data_folder)]

	if opts.split=='old':
		op_paths.sort(key=util.old_order)
		#print('train')
		train_set = load_data(op_paths[00:60],opts,data_aug=data_aug,shuffle=opts.shuffle)
		#print('val')
		val_set   = load_data(op_paths[59:60],opts,data_aug=False,shuffle=False)
		#print('test')
		test_set  = load_data(op_paths[60:80],opts,data_aug=False,shuffle=False)
	elif opts.split=='tecno':
		op_paths.sort(key=os.path.basename)
		#print('train')
		train_set = load_data(op_paths[00:40],opts,data_aug=data_aug,shuffle=opts.shuffle)
		#print('val')
		val_set   = load_data(op_paths[40:48],opts,data_aug=False,shuffle=False)
		#print('test')
		test_set  = load_data(op_paths[48:80],opts,data_aug=False,shuffle=False)
	elif opts.split=='cuhk':
		op_paths.sort(key=os.path.basename)
		#print('train')
		train_set = load_data(op_paths[00:32],opts,data_aug=data_aug,shuffle=opts.shuffle)
		#print('val')
		val_set   = load_data(op_paths[32:40],opts,data_aug=False,shuffle=False)
		#print('test')
		test_set  = load_data(op_paths[40:80],opts,data_aug=False,shuffle=False)
	elif opts.split=='cuhknotest':
		op_paths.sort(key=os.path.basename)
		#print('train')
		train_set = load_data(op_paths[00:40],opts,data_aug=data_aug,shuffle=opts.shuffle)
		#print('val')
		val_set   = load_data(op_paths[32:40],opts,data_aug=False,shuffle=False)
		#print('test')
		test_set  = load_data(op_paths[40:80],opts,data_aug=False,shuffle=False)

	return train_set, val_set, test_set

def load_data(op_paths,opts,data_aug,shuffle):

	if shuffle:

		data = []
		for op_path in op_paths:
			ID = os.path.basename(op_path)
			if os.path.isdir(op_path):
				dataset = Cholec80Test(op_path, ID, opts, data_aug, opts.seq_len)
				data.append(dataset)
		data = ConcatDataset(data)
		data = DataLoader(data, batch_size=opts.batch_size, shuffle=True, num_workers=opts.workers)
		data = [('shuffled',data)]

	else:

		data = []
		for op_path in op_paths:
			ID = os.path.basename(op_path)
			if os.path.isdir(op_path):
				#print(ID)
				dataset = Cholec80Test(op_path, ID, opts, data_aug, seq_len=1)
				dataloader = DataLoader(dataset, batch_size=opts.batch_size*opts.seq_len, shuffle=False, num_workers=opts.workers, collate_fn=collate_noshuffle)
				data.append((ID,dataloader))

	return data

# during carry hidden training or evaluation, shuffling in the dataloader is turned off
# i.e. the dataloader provides consecutive video frames along the batch dimension,
# so batch dim is actually the tmeporal dim and has tobe swapped
def collate_noshuffle(batch):

	# in the unshuffled case, "default_collate" loads video batches as Tx1xCxHxW and targets as Tx1
	data = torch.stack([b[0] for b in batch])
	target = torch.stack([b[1] for b in batch])
	data = data.transpose(0,1) # transpose to 1xTxCxHxW
	target = target.transpose(0,1) # transpose to 1xT

	return data, target

def prepare_image_features(net,opts,test_mode=False):

	train_set, val_set, test_set = prepare_dataset(opts)

	if test_mode:
		print('extracting  test features...')
		test_set = extract_features(test_set,net)
		return None,None,test_set
	else:
		print('extracting train features...')
		train_set = extract_features(train_set,net)
		print('extracting   val features...')
		val_set = extract_features(val_set,net)
		print('extracting  test features...')
		test_set = extract_features(test_set,net)
		return train_set, val_set, test_set

def extract_features(dataset,net):

	with torch.no_grad():
		net.eval()
		temp_dataset = []
		for ID,op in dataset:
			features, labels = [],[]
			for data, target in op:
				data = data.cuda()
				target = target
				feat = net.extract_image_features(data)
				features.append(feat.cpu())
				labels.append(target)
				#break
			features = torch.cat(features,dim=1)
			labels = torch.cat(labels,dim=1)
			temp_dataset.append((ID,[(features,labels)]))
			#break
		net.train()
		return temp_dataset

def prepare_batch(data,target):

	data, target = data.clone(), target.clone() # cloning prevents target from being altered during 2-step training
	data, target = data.cuda(), target.cuda()

	return data, target

class Cholec80Test():
	def __init__(self, image_path, ID, opts, seq_len=None):
		# set_trace()
		#print(ID)
		self.opts = opts
		self.image_path = image_path
		self.seq_len = opts.seq_len if seq_len is None else seq_len
		self.ext = 'jpg'
		self.dynamic_infer = seq_len is not None

		self.transform = A.Compose([
			A.SmallestMaxSize(max_size=opts.height),
			A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
			ToTensorV2(),
		])

		# load phase ground truth
		if opts.task == 'phase':
			phase_map = {
				'Preparation':0,
				'CalotTriangleDissection':1,
				'ClippingCutting':2,
				'GallbladderDissection':3,
				'GallbladderPackaging':4,
				'CleaningCoagulation':5,
				'GallbladderRetraction':6
			}
			annotation_path = os.path.join(opts.annotation_folder,'phase_annotations',"video"+ID+"-phase.txt")
			with open(annotation_path, "r") as f:
				self.target = []
				reader = csv.reader(f, delimiter='\t')
				next(reader, None)
				for count,row in enumerate(reader):
					if count % 25 == 0:
						self.target.append(phase_map[row[1]])
			self.target = torch.LongTensor(self.target)

		self.idx = 0
		self.imagebank = []
		self.targetbank = []
  
	def __next__(self):
		# print('this is being called next line 264: ',self.idx)
		img, target = self.load_frame(self.idx,self.opts)
		# if self.idx == 0:
		# 	self.imagebank = [img] * self.seq_len
		# 	self.targetbank = [target] * self.seq_len
		if not self.dynamic_infer:
			if self.idx < self.seq_len:
				self.imagebank.append(img)
				self.targetbank.append(target)
			else:
				self.imagebank = self.imagebank[1:] + [img]
				self.targetbank = self.targetbank[1:] + [target]
			img_seq, target_seq = torch.stack(self.imagebank).unsqueeze(0), torch.stack(self.targetbank).unsqueeze(0)
		
		else:
		
			img_seq, target_seq = torch.stack([img]).unsqueeze(0), torch.stack([target]).unsqueeze(0)
		self.idx += 1

		return img_seq, target_seq

	def load_frame(self,index,opts):
		# set_trace()
		target = self.target[index]
		# set_trace()
		

		file_name = os.path.join(self.image_path,'{:08d}.{}'.format(index,self.ext))
		img = cv2.imread(file_name, cv2.IMREAD_COLOR)
		# print(f'image type :{type(img)}')
		# print(f'image shape: {img.shape}')
		img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)    # still NumPy
		assert img is not None, f"Error: Could not load image at {file_name}"

		# Convert to float32 Torch tensor of shape [C, H, W]:
		img = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
		# print(f'image type after torch tensor: {type(img)}')

		# if img.dim() == 3:  # Check if in test mode
		# 	img = img.unsqueeze(0)  # Convert [C, H, W] → [1, C, H, W] (Batch dimension)
		# 	print(f"✅ Adjusted test image shape to {img.shape}") 

		 # **Before corruptions, remove batch dimension if present**
		img = img.squeeze(0)
	
		# if opts.corruption_type == 'gaussian_noise':
		# 	img = corruptions.add_gaussian_noise(img)
		# elif opts.corruption_type == 'motion_blur':
		# 	img = corruptions.apply_motion_blur(img)
		# elif opts.corruption_type == 'defocus_blur':
		# 	img = corruptions.apply_defocus_blur(img)
		# elif opts.corruption_type == 'uneven_illumination':
		# 	img = corruptions.uneven_illumination(img)
		# elif opts.corruption_type == 'smoke_effect':
		# 	img = corruptions.add_smoke_effect(img, intensity=0.7)
		# elif opts.corruption_type == 'random':
		# 	img = corruptions.random_corrupt(img)
		# else:
		# 	img = img
		
		# Albumentations typically wants a NumPy image in [H, W, C] order.
		# img = img.permute(1, 2, 0).cpu().numpy().astype('uint8')
		# print(f'image type after numpy: {type(img)}')

		#convert back to NumPy 
		img = img.permute(1, 2, 0).cpu().numpy()  # Convert to [H, W, C]
		img = (img * 255).astype(np.uint8)  # Ensure dtype is uint8

		img = self.transform(image=img)['image']

		return img, target

	def __len__(self):
		return len(self.target)

