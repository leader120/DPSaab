import pickle
import numpy as np
import MNIST_FF.data as data
import MNIST_FF.saab as saab
import time
import keras
import sklearn

def main():
	t0 = time.time()
	# load data
	fr=open('pca_params.pkl','rb')  
	pca_params=pickle.load(fr, encoding='latin')
	fr.close()

	# read data
	train_images, train_labels, test_images, test_labels, class_list = data.import_data("0-9")
	print('Training image size:', train_images.shape)
	print('Testing_image size:', test_images.shape)
	
	# Training
	print('--------Training--------')
	feature=saab.initialize(train_images, pca_params) 
	feature=feature.reshape(feature.shape[0],-1)
	print("S4 shape:", feature.shape)
	print('--------Finish Feature Extraction subnet--------')
	print("------------------- End: getfeature -> using %10f seconds" % (time.time() - t0))
	feat={}
	feat['feature']=feature
	'''batch_size = 1000
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
	feat['feature'] = feature'''
	
	# save data
	fw=open('feat.pkl','wb')    
	pickle.dump(feat, fw)    
	fw.close()

if __name__ == '__main__':
	main()
