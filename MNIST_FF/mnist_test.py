import MNIST_FF.data as data
import MNIST_FF.saab as saab
import pickle
import numpy as np
import sklearn
import time
#import cv2
import keras
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from numpy import linalg as LA
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import euclidean_distances
#from mlxtend.evaluate import confusion_matrix
import matplotlib.pyplot as plt
#from mlxtend.plotting import plot_confusion_matrix

def main():
	t0=time.time()
	# load data
	fr=open('pca_params.pkl','rb')  
	pca_params=pickle.load(fr, encoding='latin')
	fr.close()



	# read data
	train_images, train_labels, test_images, test_labels, class_list = data.import_data("0-9")
	print('Training image size:', train_images.shape)
	print('Testing_image size:', test_images.shape)
	'''batch_size = 2000
	num_samples = int(len(test_images) / batch_size)

	print('--------Testing--------')
	features = []
	for i in range(num_samples):
		trn_images = test_images[i * batch_size:i * batch_size + batch_size, :]
		feature = saab.initialize(trn_images, pca_params)
		feature = feature.reshape(feature.shape[0], -1)
		features.append(feature)
	feature = np.vstack(features)'''
	print('--------Testing--------')
	feature = saab.initialize(test_images, pca_params)
	feature = feature.reshape(feature.shape[0], -1)
	print("S4 shape:", feature.shape)
	print('--------Finish Feature Extraction subnet--------')




	# feature normalization
	std_var=(np.std(feature, axis=0)).reshape(1,-1)
	feature=feature/std_var
	
	num_clusters=[120,84, 10]
	use_classes=10
	fr=open('llsr_weights.pkl','rb')  
	weights=pickle.load(fr)
	fr.close()
	fr=open('llsr_bias.pkl','rb')  
	biases=pickle.load(fr)
	fr.close()

	for k in range(len(num_clusters)):
		# least square regression
		weight=weights['%d LLSR weight'%k]
		bias=biases['%d LLSR bias'%k]
		feature=np.matmul(feature,weight)
		feature=feature+bias
		print(k,' layer LSR weight shape:', weight.shape)
		print(k,' layer LSR bias shape:', bias.shape)
		print(k,' layer LSR output shape:', feature.shape)
		
		if k!=len(num_clusters)-1:
			pred_labels=np.argmax(feature, axis=1)
			num_clas=np.zeros((num_clusters[k],use_classes))
			for i in range(num_clusters[k]):
				for t in range(use_classes):
					for j in range(feature.shape[0]):
						if pred_labels[j]==i and test_labels[j]==t:
							num_clas[i,t]+=1
			acc_test=np.sum(np.amax(num_clas, axis=1))/feature.shape[0]
			print(k,' layer LSR testing acc is {}'.format(acc_test))

			# Relu
			for i in range(feature.shape[0]):
				for j in range(feature.shape[1]):
					if feature[i,j]<0:
						feature[i,j]=0

		else:
			pred_labels=np.argmax(feature, axis=1)
			acc_test=sklearn.metrics.accuracy_score(test_labels,pred_labels)
			f_measurse=sklearn.metrics.f1_score(test_labels,pred_labels,average='weighted')
			report=sklearn.metrics.classification_report(pred_labels,test_labels)
			print(report)
			print('f1 is {}'.format(f_measurse))
			print('testing acc is {}'.format(acc_test))
			print("------------------- End: test -> using %10f seconds" % (time.time() - t0))
	'''confusion = confusion_matrix(y_target=test_labels, y_predicted=pred_labels, binary=False)
	fig, ax = plot_confusion_matrix(conf_mat=confusion)
	plt.show()'''

if __name__ == '__main__':
	main()
'''
	# testing
	print('--------Testing--------')
	feature=saab.initialize(test_images, pca_params)
	feature=feature.reshape(feature.shape[0],-1)
	print("S4 shape:", feature.shape)
	print('--------Finish Feature Extraction subnet--------')'''

