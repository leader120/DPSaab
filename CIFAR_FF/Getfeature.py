import pickle
import numpy as np
import CIFAR_FF.data as data
import CIFAR_FF.saab as saab
import time
import keras
import sklearn
import matplotlib.pyplot as plt

def main():
	t0=time.time()
	# load data
	fr=open('pca_params.pkl','rb')  
	pca_params=pickle.load(fr, encoding='latin1')
	fr.close()

	# read data
	train_images, train_labels, test_images, test_labels, class_list = data.import_data("0-9")
	print('Training image size:', train_images.shape)
	print('Testing_image size:', test_images.shape)

	feat={}
	# Training
	print('--------Training--------')
	feature=saab.initialize(train_images, pca_params) 
	print("S4 shape:", feature.shape)
	print('--------Finish Feature Extraction subnet--------')
	feat['training_feature']=feature

	print('--------Testing--------')
	feature=saab.initialize(test_images, pca_params) 
	print("S4 shape:", feature.shape)
	print('--------Finish Feature Extraction subnet--------')
	feat['testing_feature']=feature
	'''batch_size = 5000
	num_samples = int(len(train_images) / batch_size)

	# Training
	print('--------Training--------')
	features = []
	for i in range(num_samples):
		trn_images = train_images[i * batch_size:i * batch_size + batch_size, :]
		feature = saab.initialize(trn_images, pca_params)
		feature = feature.reshape(feature.shape[0], -1)
		features.append(feature)
	feature = np.vstack(features)
	print("S4 shape:", feature.shape)
	print('--------Finish Feature Extraction subnet--------')
	feat = {}
	feat['training_feature'] = feature
	print("------------------- End: getfeature -> using %10f seconds" % (time.time() - t0))'''

	'''print('--------Testing--------')
	features = []
	for i in range(num_samples):
		trn_images = test_images[i * batch_size:i * batch_size + batch_size, :]
		feature = saab.initialize(trn_images, pca_params)
		feature = feature.reshape(feature.shape[0], -1)
		features.append(feature)
	feature = np.vstack(features)
	print("S4 shape:", feature.shape)
	print('--------Finish Feature Extraction subnet--------')
	feat = {}
	feat['testing_feature'] = feature'''

	# save data
	fw=open('feat.pkl','wb')    
	pickle.dump(feat, fw)    
	fw.close()


if __name__ == '__main__':
	main()
