import pickle
import keras
import CIFAR_FF.data as data
from keras.datasets import cifar10
import numpy as np
import sklearn
import CIFAR_FF.saab as saab
import time
#import cv2
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from numpy import linalg as LA
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import euclidean_distances

def main():
	t0=time.time()
	fr = open('pca_params.pkl', 'rb')
	pca_params = pickle.load(fr, encoding='latin')
	fr.close()

	fr=open('llsr_weights.pkl','rb')  
	weights=pickle.load(fr, encoding='latin1')
	fr.close()
	fr=open('llsr_bias.pkl','rb')  
	biases=pickle.load(fr, encoding='latin1')
	fr.close()

	# load feature
	fr = open('feat.pkl', 'rb')
	feat = pickle.load(fr, encoding='latin1')
	fr.close()
	feature = feat['testing_feature']
	feature = feature.reshape(feature.shape[0], -1)
	print("S4 shape:", feature.shape)
	print('--------Finish Feature Extraction subnet--------')

	# read data
	train_images, train_labels, test_images, test_labels, class_list = data.import_data("0-9")
	print('Training image size:', train_images.shape)
	print('Testing_image size:', test_images.shape)
	'''batch_size = 1000
	num_samples = int(len(test_images) / batch_size)

	print('--------Testing--------')
	features = []
	for i in range(num_samples):
		trn_images = test_images[i * batch_size:i * batch_size + batch_size, :]
		feature = saab.initialize(trn_images, pca_params)
		feature = feature.reshape(feature.shape[0], -1)
		features.append(feature)
	feature = np.vstack(features)
	print("S4 shape:", feature.shape)
	print('--------Finish Feature Extraction subnet--------')'''




	# feature normalization
	#std_var=(np.std(feature, axis=0)).reshape(1,-1)
	#feature=feature/std_var

	num_clusters=[200, 100, 10]
	use_classes=10
	for k in range(len(num_clusters)):
		weight=weights['%d LLSR weight'%k]
		bias=biases['%d LLSR bias'%k]
		feature=np.matmul(feature,weight)+bias
		print(k,' layer LSR weight shape:', weight.shape)
		print(k,' layer LSR output shape:', feature.shape)
		if k!=len(num_clusters)-1:
			# Relu
			for i in range(feature.shape[0]):
				for j in range(feature.shape[1]):
					if feature[i,j]<0:
						feature[i,j]=0
		else:
			pred_labels=np.argmax(feature, axis=1)
			acc_test=sklearn.metrics.accuracy_score(test_labels,pred_labels)
			print('testing acc is {}'.format(acc_test))
			print("------------------- End: getfeature -> using %10f seconds" % (time.time() - t0))
if __name__ == '__main__':
	main()


