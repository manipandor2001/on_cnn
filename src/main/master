#OC_CNN

import torch
import torchvision


import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as func
import torchvision.models as models
import torchvision.transforms as transforms


import cv2
import scipy.io
from scipy import misc
from sklearn import svm
from sklearn import metrics
import joblib
from sklearn.metrics import accuracy_score


import os
import sys
import copy
import h5py
import time
import pickle
import random
import argparse

import numpy as np
#import p
# import matplotlib.pyplot as plt



import numpy as np

### Setting hyperparameters
class hyperparameters():
	def __init__(self):
		self.batch_size                  = 64
		self.iterations                  = 1000
		self.lr							 = 1e-4
		self.sigma 						 = 0.01
		self.sigma1						 = 0.00000000000000000000000000000001
		self.D                           = 4096
		self.N 							 = 0.5
		self.gamma 						 = float(1/4096.0)

		self.stats_freq 				 = 1
		self.img_chnl 					 = 3
		self.img_size 					 = 224

		self.gpu_flag 					 = True
		self.verbose 					 = False
		self.pre_trained_flag			 = True
		self.intensity_normalization	 = False

		self.model_type 				 = 'alexnet'
		self.method 					 = 'OC-CNN'
		self.classifier_type			 = 'OC-CNN'



### Setting print colors
class bcolors:
	HEADER	  = '\033[95m'
	BLUE	  = '\033[94m'
	GREEN	  = '\033[92m'
	YELLOW 	  = '\033[93m'
	FAIL	  = '\033[91m'
	ENDC	  = '\033[0m'
	BOLD      = '\033[1m'
	UNDERLINE = '\033[4m'



hyper_para = hyperparameters()
colors     = bcolors()

def AddNoise(inputs, sigma):

	noise_shape = np.shape(inputs)
	
	noise = np.random.normal(0, sigma, noise_shape)
	noise = torch.from_numpy(noise)
	noise = (noise).float()

	if(inputs.is_cuda):
		outputs = inputs + noise.cuda()
	else:
		outputs = inputs + noise

	return outputs

def get_fuv(hyper_para, model_type):

	## defining frequnetly used global variables
	running_loss=0.0

	inm  = nn.InstanceNorm1d(1, affine=False)
	relu = nn.ReLU()

	mean = 0.0*np.ones( (hyper_para.D,) )
	cov  = hyper_para.sigma*np.identity(hyper_para.D)

	if(model_type=='vggface'):
		imagenet_mean = np.asarray([0.485, 0.456, 0.406])
		imagenet_std  = np.asarray([0.229, 0.224, 0.225])
	else:
		imagenet_mean = np.asarray([0.36703529411, 0.410832941, 0.506612941])
		imagenet_std  = np.asarray([1.0, 1.0, 1.0])

	classifier = classifier_nn(hyper_para.D)

	return running_loss, inm, relu, mean, cov, imagenet_mean, imagenet_std, classifier


def choose_network(model_type, pre_trained_flag, dataset=None):
    if model_type == 'alexnet':
        model = models.alexnet(pretrained=pre_trained_flag)

        # Modify the first convolution layer if the dataset is MNIST (grayscale input)
        
        model.features[0] = nn.Conv2d(1, 64, kernel_size=2, stride=2, padding=1)  # for grayscale images like MNIST

        # Remove the last two layers from the classifier
        new_classifier = nn.Sequential(*list(model.classifier.children())[:-2])
        model.classifier = new_classifier

    elif model_type == 'vgg16':
        model = torchvision.models.vgg16(pretrained=pre_trained_flag)
        new_classifier = nn.Sequential(*list(model.classifier.children())[:-3])
        model.classifier = new_classifier

    elif model_type == 'vgg19':
        model = torchvision.models.vgg19(pretrained=pre_trained_flag)
        new_classifier = nn.Sequential(*list(model.classifier.children())[:-3])
        model.classifier = new_classifier

    elif model_type == 'vgg16bn':
        model = torchvision.models.vgg16_bn(pretrained=pre_trained_flag)
        new_classifier = nn.Sequential(*list(model.classifier.children())[:-3])
        model.classifier = new_classifier

    elif model_type == 'vgg19bn':
        model = torchvision.models.vgg19_bn(pretrained=pre_trained_flag)
        new_classifier = nn.Sequential(*list(model.classifier.children())[:-3])
        model.classifier = new_classifier

    elif model_type == 'vggface':
        model = VGG_FACE_torch.VGG_FACE_torch
        model.load_state_dict(torch.load('VGG_FACE.pth'))
        model = model[:-3]

    else:
        raise argparse.ArgumentTypeError('Supported models are alexnet, vgg16, vgg19, vgg16bn, vgg19bn. \n Enter model_type as one of these arguments.')

    return model

def choose_classifier(dataset, class_number, model_type, model, classifier, D, hyper_para, train_data, test_data, test_label, no_train_data, no_test_data, inm, relu, m, s):

	if(hyper_para.verbose==True):
		print('Extracting features.....')

	train_features = np.memmap('../../temp_files/train_features_temp.bin', dtype='float32', mode='w+', shape=(no_train_data,hyper_para.D))
	train_features = torch.from_numpy(train_features)

	for i in range(no_train_data):
		temp = model((train_data[i:(i+1)].cuda().contiguous().float())).float()
		temp = temp.view(1,1,hyper_para.D)
		temp = inm(temp)
		temp = relu(temp.view(hyper_para.D))
		train_features[i:(i+1)] = temp.data.cpu()
	train_data = None

	if(hyper_para.verbose==True):
		print('Features extracted.')

	## test on the test set
	#test_features = np.memmap('../../temp_files/test_features_temp.bin', dtype='float32', mode='w+', shape=(no_test_data,hyper_para.D))
	#test_scores   = np.memmap('../../temp_files/test_scores_temp.bin', dtype='float32', mode='w+', shape=(no_test_data,1))
	test_features = torch.from_numpy(test_features)

	if(hyper_para.verbose==True):
		print('Computing test scores and AUC....')

	area_under_curve=0.0
	if(hyper_para.classifier_type=='OC_CNN'):
		test_scores   = torch.from_numpy(test_scores)
		k=0
		print(np.shape(test_features))
		start = time.time()
		for j in range(no_test_data):
			temp = model(AddNoise((test_data[j:(j+1)].cuda().contiguous().float()), hyper_para.sigma1)).float()
			temp = temp.view(1,1,hyper_para.D)
			temp = inm(temp)
			temp = temp.view(hyper_para.D)
			
			test_features[k:(k+1)] = temp.data.cpu()
			test_scores[k:(k+1)]   = classifier(relu(temp)).data.cpu()[1]
			# print(classifier(relu(temp)).data.cpu())
			
			k = k+1
		end = time.time()
		print(end-start)
		test_scores    = test_scores.numpy()
		test_features  = test_features.numpy()
		train_features = train_features.numpy()

		test_scores = (test_scores-np.min(test_scores))/(np.max(test_scores)-np.min(test_scores))

	elif(hyper_para.classifier_type=='OC_SVM_linear'):
		# train one-class svm
		oc_svm_clf = svm.OneClassSVM(kernel='linear', nu=float(hyper_para.N))
		oc_svm_clf.fit(train_features.numpy())
		k=0
		mean_kwn = np.zeros( (no_test_data,1) )
		for j in range(no_test_data):
			temp = model((test_data[j:(j+1)].cuda().contiguous().float())).float()
			temp = temp.view(1,1,hyper_para.D)
			temp = inm(temp)
			temp = temp.view(hyper_para.D)			
			test_features[k:(k+1)] = temp.data.cpu()
			temp 				   = np.reshape(relu(temp).data.cpu().numpy(), (1, hyper_para.D))
			test_scores[k:(k+1)]   = oc_svm_clf.decision_function(temp)[0]

			k = k+1

		test_features  = test_features.numpy()
		train_features = train_features.numpy()

		joblib.dump(oc_svm_clf,'../../save_folder/saved_models/'+dataset+'/classifier/'+str(class_number) +'/'+
																				model_type+'_OCCNNlin'    +'_'+
																				str(hyper_para.iterations)+'_'+
																				str(hyper_para.lr)		  +'_'+
																				str(hyper_para.sigma)	  +'_'+
																				str(hyper_para.N)         +'.pkl')

	fpr, tpr, thresholds = metrics.roc_curve(test_label, test_scores)

	if(hyper_para.verbose==True):
		print('Test scores and AUC computed.')

	return area_under_curve, train_features, test_scores, test_features

def choose_method(dataset, model_type, class_number, hyper_para):

	auc=0.0
	if(hyper_para.method=='OC_CNN' or 'OC-CNN'):
		auc = OC_CNN(dataset, model_type, class_number, hyper_para)
	elif(hyper_para.method=='OC_SVM_linear'):
		auc = OC_SVM_linear(dataset, model_type, class_number, hyper_para)
	elif(hyper_para.method=='Bi_SVM_linear'):
		auc = Bi_SVM_linear(dataset, model_type, class_number, hyper_para)
	elif(hyper_para.method=='SVDD'):
		print('look at matlab code')
	elif(hyper_para.method=='SMPM'):
		print('look at matlab code')
	else:
		raise argparse.ArgumentTypeError('model_type argument can be only one of these OC_CNN, OC_SVM_linear, Bi_SVM_linear')

	return auc

def OC_CNN(dataset, model_type, class_number, hyper_para):

	running_loss, inm, relu, mean, cov, imagenet_mean, imagenet_std, classifier = get_fuv(hyper_para, model_type)

	if(hyper_para.verbose==True):
		print('Loading dataset '+dataset+'...')
  
	train_data, test_data, test_label, no_train_data, no_test_data = load_mnist_dataset()
	#train_data, test_data, test_label = load_dataset(dataset, class_number, imagenet_mean, imagenet_std, hyper_para)

	if(hyper_para.verbose==True):
		print(dataset+' dataset loaded.')


	### choose one network which produces D dimensional features
	if(hyper_para.verbose==True):
		print('Loading network '+hyper_para.model_type+'...')
	
	model = choose_network(model_type, hyper_para.pre_trained_flag)

	if(hyper_para.verbose==True):
		print('Network '+hyper_para.model_type+' loaded.')

	running_cc = 0.0
	running_ls = 0.0

	if(hyper_para.gpu_flag):
		inm.cuda()
		relu.cuda()
		model.cuda()
		classifier.cuda()
	
	model.train()
	classifier.train()
	
	### optimizer for model training (for this work we restrict to only fine-tuning FC layers)
	if(model_type=='vggface'):
		model_optimizer      = optim.Adam(model[-5:].parameters(), lr=hyper_para.lr)
	else:
		model_optimizer      = optim.Adam(model.classifier.parameters(), lr=hyper_para.lr)
	classifier_optimizer = optim.Adam(classifier.parameters(), lr=hyper_para.lr)
	
	# loss functions
	cross_entropy_criterion = nn.CrossEntropyLoss()

	for i in range(int(hyper_para.iterations)):
	# for i in range(int(hyper_para.iterations*no_train_data/hyper_para.batch_size)):
		# print i
		rand_id = np.asarray(random.sample( range(no_train_data), int(hyper_para.batch_size)))
		rand_id = torch.from_numpy(rand_id)

		# get the inputs
		inputs = train_data[rand_id].cuda().float()
		
		# get the labels
		labels = np.concatenate( (np.zeros( (int(hyper_para.batch_size),) ), np.ones( (int(hyper_para.batch_size),)) ), axis=0)
		labels = torch.from_numpy(labels)
		labels = labels.cuda().long()
		
		gaussian_data = np.random.normal(0, hyper_para.sigma, (int(hyper_para.batch_size), hyper_para.D))
		gaussian_data = torch.from_numpy(gaussian_data)

		# forward + backward + optimize
		out1 = model(AddNoise(inputs, hyper_para.sigma1))

		out1 = out1.view(int(hyper_para.batch_size), 1, hyper_para.D)
		out1 = inm(out1)
		out1 = out1.view(int(hyper_para.batch_size), hyper_para.D)
		out2 = gaussian_data.cuda().float()
		out  = torch.cat( (out1, out2),0)
		out  = relu(out)
		out  = classifier(out)
		
		# zero the parameter gradients
		model_optimizer.zero_grad()
		classifier_optimizer.zero_grad()
		 
		cc = cross_entropy_criterion(out, labels) 
		loss = cc
		
		loss.backward()

		model_optimizer.step()
		classifier_optimizer.step()
		
		# print statistics
		running_cc += cc.data
		running_loss += loss.data

		if(hyper_para.verbose==True):
			if (i % (hyper_para.stats_freq) == (hyper_para.stats_freq-1)):    # print every stats_frequency batches
				line = hyper_para.BLUE   + '[' + str(format(i+1, '8d')) + '/'+ str(format(int(hyper_para.iterations), '8d')) + ']' + hyper_para.ENDC + \
					hyper_para.GREEN  + ' loss: '     + hyper_para.ENDC + str(format(running_loss/hyper_para.stats_freq, '1.8f'))  + \
					hyper_para.GREEN  + ' cc: '     + hyper_para.ENDC + str(format(running_cc/hyper_para.stats_freq, '1.8f'))
				print(line)
				running_loss = 0.0
				running_cc = 0.0
			
	classifier.eval()
	model.eval()
	relu.eval()

	area_under_curve, train_features, test_scores, test_features = choose_classifier(dataset, class_number, model_type, model, classifier, hyper_para.D, hyper_para, train_data, test_data, test_label, no_train_data, no_test_data, inm, relu, imagenet_mean, imagenet_std)

	classifier.cpu()
	model.cpu()
	relu.cpu()
	
	torch.save(model,'../../save_folder/saved_models/'+dataset+'/model/'+str(class_number)+'/'+model_type +'_'+
																				str(hyper_para.iterations)+'_'+
																				str(hyper_para.lr)		  +'_'+
																				str(hyper_para.sigma)	  +'.pth')
	
	torch.save(model,'../../save_folder/saved_models/'+dataset+'/classifier/'+str(class_number)+'/'+model_type +'_'+
																					 str(hyper_para.iterations)+'_'+
																					 str(hyper_para.lr)		   +'_'+
																					 str(hyper_para.sigma)     +'.pth')

	scipy.io.savemat('../../save_folder/results/'+dataset+'/'+ str(class_number)  +'/'+ model_type	+'_OCCNN123_'+
							 str(hyper_para.iterations)  +'_'+ str(hyper_para.lr) +'_'+ str(hyper_para.sigma)	 +'.mat',
								{'auc':area_under_curve, 'train_features':train_features, 'test_scores':test_scores,
														 'test_features':test_features,   'test_label':test_label    })

	if(hyper_para.verbose==True):
		print('model, , featureclassifiers and results saved.')

	return area_under_curve




HYPER_PARA = hyperparameters()
