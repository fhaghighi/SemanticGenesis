# Copyright (c) ASU GitHub Project.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
################################################################################


import numpy as np
from torch import nn
import torch
from torchsummary import summary
import sys
from data_loader import *
from models.ynet3d import *
from config import setup_config
from tqdm import tqdm
import os
import argparse
from sklearn.metrics import accuracy_score

parser = argparse.ArgumentParser(description='PyTorch Semantic Genesis Training')
parser.add_argument('--notes', default=None, type=str, help='which exp')
parser.add_argument("--input_rows", dest="input_rows", help="input rows", default=64, type=int)
parser.add_argument("--input_cols", dest="input_cols", help="input cols", default=64, type=int)
parser.add_argument("--input_deps", dest="input_deps", help="input deps", default=32, type=int)
parser.add_argument("--verbose", dest="verbose", help="verbose", default=1, type=int)

parser.add_argument("--cls_classes", dest="cls_classes", help="number of classes", default=44,type=int)
parser.add_argument("--nb_instances", dest="nb_instances", help="nubmer of samples in each class",type=int, default=200)
parser.add_argument("--nb_val_instances", dest="nb_val_instances", help="nubmer of validation instances in each class",type=int, default=30)
parser.add_argument("--nb_multires_patch", dest="nb_multires_patch",help="number of multi resolution cubes", type=int, default=3)
parser.add_argument("--lambda_rec", dest="lambda_rec",help="reconstruction loss weight", type=int, default=1)
parser.add_argument("--lambda_cls", dest="lambda_cls",help="classification loss weight", type=int, default=0.01)
parser.add_argument("--data_dir", dest="data_dir",help="data path", default=None)


parser.add_argument('--multi_gpu', action='store_true', default=False, help='use multi gpu?')
parser.add_argument('--batch_size',default=64, type=int, help='batch size')
parser.add_argument('--num_workers',default=4, type=int, help='number of workers')
parser.add_argument('--exp', default='en_de', type=str, help='which part include in the exp: en|en_de')
parser.add_argument('--pre_path', default=None, type=str, help='path to pre-trained model')
parser.add_argument('--lr',default=1e-3, type=float, help='learning rate')


args = parser.parse_args()

assert args.data_dir is not None


torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = False

conf = setup_config(input_rows=args.input_rows,
                      input_cols=args.input_cols,
                      input_deps=args.input_deps,
                      batch_size=args.batch_size,
                      verbose=args.verbose,
                      cls_classes=args.cls_classes,
                      nb_instances=args.nb_instances,
					  nb_val_instances = args.nb_val_instances,
                      nb_multires_patch=args.nb_multires_patch,
                      lambda_rec=args.lambda_rec,
                      lambda_cls=args.lambda_cls,
                      DATA_DIR=args.data_dir,
					exp_choice = args.exp)
conf.display()
print("torch = {}".format(torch.__version__),file=conf.log_writter)


x_train = []
y_train=[]
for i in range (int(math.ceil(conf.cls_classes/50))):
	print("data part:",i)
	for fold in range (conf.nb_multires_patch):
		print("fold:",fold)
		s = np.load(os.path.join(conf.DATA_DIR, "train_data"+str(fold+1)+"_vwGen_ex_ref_fold"+str(i+1)+".0.npy"))
		l = np.load(os.path.join(conf.DATA_DIR, "train_label"+str(fold+1)+"_vwGen_ex_ref_fold"+str(i+1)+".0.npy"))
		if (i==int(math.ceil(conf.cls_classes/50))-1) and conf.cls_classes % 50 != 0:
			print("select subset of data")
			index=conf.cls_classes - i * 50
			s=s[0:conf.nb_instances*index,:]
			l=l[0:conf.nb_instances*index,:]
		s = s[:, np.newaxis, :, :, :]
		x_train.extend(s)
		y_train.extend(l)
		del s
x_train=np.array(x_train)
y_train=np.array(y_train)


print("x_train: {} | {:.2f} ~ {:.2f}".format(x_train.shape, np.min(x_train), np.max(x_train)))
print("y_train: {} | {:.2f} ~ {:.2f}".format(y_train.shape, np.min(y_train), np.max(y_train)))

x_valid = []
y_valid=[]
for i in range (int(math.ceil(conf.cls_classes/50))):
	print("data part:",i)
	s = np.load(os.path.join(conf.DATA_DIR, "val_data1_vwGen_ex_ref_fold"+str(i+1)+".0.npy"))
	l = np.load(os.path.join(conf.DATA_DIR, "val_label1_vwGen_ex_ref_fold"+str(i+1)+".0.npy"))
	if (i == int(math.ceil(conf.cls_classes / 50)) - 1) and conf.cls_classes % 50 != 0:
		print("select subset of data")
		index = conf.cls_classes - i * 50
		s = s[0:conf.nb_val_instances * index, :]
		l = l[0:conf.nb_val_instances * index, :]
	s = s[:, np.newaxis, :, :, :]
	x_valid.extend(s)
	y_valid.extend(l)
	del s
x_valid=np.array(x_valid)
y_valid=np.array(y_valid)
print("x_valid: {} | {:.2f} ~ {:.2f}".format(x_valid.shape, np.min(x_valid), np.max(x_valid)))
print("y_valid: {} | {:.2f} ~ {:.2f}".format(y_valid.shape, np.min(y_valid), np.max(y_valid)))




training_data = SemanticGenesis_Dataset(x_train, y_train, conf)
validation_data = SemanticGenesis_Dataset(x_valid, y_valid, conf)

params = {'batch_size': args.batch_size,
          'shuffle': True,
          'num_workers': args.num_workers}
          
training_generator = torch.utils.data.DataLoader(training_data,**params)
validation_generator = torch.utils.data.DataLoader(validation_data,**params)

device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print(torch.cuda.device_count())

if conf.exp_choice == 'en':
	model = ClassificationNet3D(cls_classes=conf.cls_classes)
elif conf.exp_choice == 'en_de':
	model = YNet3D(cls_classes=conf.cls_classes)
	if args.pre_path is not None:
		checkpoint_en = torch.load(args.pre_path)
		for name in model.state_dict().keys():
			if name in checkpoint_en['state_dict'].keys():
				model.state_dict()[name].copy_(checkpoint_en['state_dict'][name])
		print("loaded model from %s" %args.pre_path)
if args.multi_gpu:
    model = nn.DataParallel(model, device_ids = [i for i in range(torch.cuda.device_count())])
model.to(device)

print("Total CUDA devices: ", torch.cuda.device_count() ,file=conf.log_writter)


criterion_mse = nn.MSELoss()
criterion_cce = nn.CrossEntropyLoss()

if conf.optimizer == "sgd":
	optimizer = torch.optim.SGD(model.parameters(), conf.lr, momentum=0.9, weight_decay=0.0, nesterov=False)
elif conf.optimizer == "Adam":
	optimizer = torch.optim.Adam(model.parameters(), conf.lr)
else:
	raise

scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=int(conf.patience * 0.8), gamma=0.5)

cce_losses = []
mse_losses = []


avg_train_losses = []
avg_valid_losses = []
best_loss = 100000
best_acc = 0
intial_epoch =0
num_epoch_no_improvement = 0
conf.log_writter.flush()

if os.path.exists(os.path.join(conf.model_path,"semantic_genesis_chest_ct.pt")):
	checkpoint=torch.load(os.path.join(conf.model_path,"semantic_genesis_chest_ct.pt"))
	model.load_state_dict(checkpoint['state_dict'])
	optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
	intial_epoch=checkpoint['epoch']
	best_loss = checkpoint['best_loss']
	num_epoch_no_improvement = checkpoint['num_epoch_no_improvement']
	print("Loading weights from ",os.path.join(conf.model_path,"semantic_genesis_chest_ct.pt"),file=conf.log_writter)


conf.log_writter.flush()


for epoch in range(intial_epoch,conf.nb_epoch):
	print(epoch)
	model.train()
	train_losses=[]
	valid_losses=[]
	
	if args.exp == 'en':
		train_acc = []
		valid_acc = []
		for iteration, (x,y) in enumerate(tqdm(training_generator)):
	
		    image, gt = x.float().to(device), y.long().to(device)
		    pred=model(image)
		    
		    optimizer.zero_grad()
		    loss = criterion_cce(torch.log(pred),gt)
		
		    loss.backward()
		    optimizer.step()
		    
		    train_acc.append(accuracy_score(y.cpu().numpy().astype('int32'), np.argmax(pred.cpu().detach().numpy(),axis=1)))
		    
		    train_losses.append(round(loss.item(), 2))
		    if (iteration + 1) % 5 ==0:
		        print('Epoch [{}/{}], iteration [{}/{}], cross_entropy Loss: {:.6f}, train_acc: {:.6f},'
		        .format(epoch + 1, conf.nb_epoch, iteration + 1, int(x_train.shape[0]//args.batch_size) , 
		        np.average(train_losses), np.average(train_acc)) ,file=conf.log_writter)
		        conf.log_writter.flush()
		    torch.cuda.empty_cache()
		with torch.no_grad():
		    model.eval()
		    print("validating....",file=conf.log_writter)
		    for i, (x,y) in enumerate(tqdm(validation_generator)):
			    image,gt = x.float().to(device), y.long().to(device)
			    pred=model(image)
			    loss = criterion_cce(torch.log(pred),gt)
			    valid_losses.append(loss.item())

			    valid_acc.append(accuracy_score(y.cpu().numpy().astype('int32'), np.argmax(pred.cpu().detach().numpy(),axis=1)))
				
			    if (i + 1) % 5 ==0:
			        print('Validation Epoch [{}/{}], iteration [{}/{}]'
					    .format(epoch + 1, conf.nb_epoch, i + 1, int(x_valid.shape[0]//args.batch_size) ) ,file=conf.log_writter)
			        conf.log_writter.flush()
	
	if args.exp == 'en_de':
		valid_acc = []
		for iteration, (x,[y_img,y_clc,y_trans]) in enumerate(tqdm(training_generator)):
			image, gt_img, gt_clc = x.float().to(device), y_img.float().to(device), y_clc.long().to(device)
			pred_img, pred_clc = model(image)
			
			optimizer.zero_grad()
			
			cce_loss = conf.lambda_cls * criterion_cce(torch.log(pred_clc),gt_clc)
			mse_loss = conf.lambda_rec * criterion_mse(pred_img,gt_img)

			loss = cce_loss + mse_loss
			loss.backward()
			optimizer.step()
			
			cce_losses.append(round(cce_loss.item(), 2))
			mse_losses.append(round(mse_loss.item(), 2))
			train_losses.append(round(loss.item(), 2))
			if (iteration + 1) % 5 ==0:
				print('Epoch [{}/{}], iteration [{}/{}], cce loss: {:.6f}, mse loss: {:.6f}, total loss: {:.6f}'
				.format(epoch + 1, conf.nb_epoch, iteration + 1, int(x_train.shape[0]//args.batch_size), np.average(cce_losses), np.average(mse_losses), np.average(train_losses)) ,file=conf.log_writter)
			conf.log_writter.flush()
		        
		torch.cuda.empty_cache()
		with torch.no_grad():
			model.eval()
			print("validating....",file=conf.log_writter)
			for i, (x,[y_img,y_clc,y_trans]) in enumerate(tqdm(validation_generator)):
				image, gt_img, gt_clc = x.float().to(device), y_img.float().to(device), y_clc.long().to(device)
				pred_img, pred_clc = model(image)
				loss = conf.lambda_cls * criterion_cce(torch.log(pred_clc),gt_clc) + conf.lambda_rec * criterion_mse(pred_img,gt_img)
				valid_losses.append(loss.item())
				valid_acc.append(accuracy_score(y_clc.cpu().numpy().astype('int32'), np.argmax(pred_clc.cpu().detach().numpy(),axis=1)))
				if (i + 1) % 5 ==0:
					print('Validation Epoch [{}/{}], iteration [{}/{}]'.format(epoch + 1, conf.nb_epoch, i + 1, int(x_valid.shape[0]//args.batch_size) ) ,file=conf.log_writter)
		conf.log_writter.flush()


	#logging
	train_loss=np.average(train_losses)
	valid_loss=np.average(valid_losses)
	validation_acc=np.average(valid_acc)
	avg_train_losses.append(train_loss)
	avg_valid_losses.append(valid_loss)
	print("Epoch {}, best acc is {:.4f}, valid acc is {:.4f}".format(epoch+1,best_acc,validation_acc),file=conf.log_writter)
	print("Epoch {}, validation loss is {:.4f}, training loss is {:.4f}".format(epoch+1,valid_loss,train_loss),file=conf.log_writter)

	if  valid_loss < best_loss:
		print("Validation  loss decreases from {:.4f} to {:.4f}".format(best_loss, valid_loss),file=conf.log_writter)
		best_acc = validation_acc
		best_loss = valid_loss
		num_epoch_no_improvement = 0
		torch.save({
			'epoch': epoch + 1,
			'state_dict': model.state_dict(),
			'optimizer_state_dict': optimizer.state_dict(),
			'num_epoch_no_improvement': num_epoch_no_improvement,
			'best_loss': best_loss,
			'best_acc': best_acc
		}, os.path.join(conf.model_path, "semantic_genesis_chest_ct.pt"))
		print("Saving model ", os.path.join(conf.model_path, "semantic_genesis_chest_ct.pt"),
			  file=conf.log_writter)


	else:
		print("Validation custom loss does not decrease from {:.4f}, num_epoch_no_improvement {}".format(best_loss,num_epoch_no_improvement),file=conf.log_writter)
		num_epoch_no_improvement += 1
	if num_epoch_no_improvement == conf.patience:
		print("Early Stopping",file=conf.log_writter)
		break
	conf.log_writter.flush()
	scheduler.step()

