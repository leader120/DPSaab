import numpy as np
from skimage.util.shape import view_as_windows
import math
import sklearn
from sklearn.decomposition import PCA
from numpy import linalg as LA
from skimage.measure import block_reduce
from sklearn.cluster import KMeans
import pickle
import matplotlib.pyplot as plt

def parse_list_string(list_string):
	"""Convert the class string to list."""
	elem_groups=list_string.split(",")
	results=[]
	for group in elem_groups:
		term=group.split("-")
		if len(term)==1:
	  		results.append(int(term[0]))
		else:
	  		start=int(term[0])
	  		end=int(term[1])
	  		results+=range(start, end+1)
	return results

# convert responses to patches representation
def window_process(samples, kernel_size, stride):
	'''
	Create patches
	:param samples: [num_samples, feature_height, feature_width, feature_channel]
	:param kernel_size: int i.e. patch size
	:param stride: int
	:return patches: flattened, [num_samples, output_h, output_w, feature_channel*kernel_size^2]

	'''
	n, h, w, c = samples.shape
	print(c)
	output_h = (h - kernel_size)//stride + 1
	output_w = (w - kernel_size)//stride + 1
	patches = view_as_windows(samples, (1, kernel_size, kernel_size, c), step=(1, stride, stride, c))
	print(patches.shape)
	patches = patches.reshape(n, output_h, output_w, c*kernel_size*kernel_size)
	return patches

def remove_mean(features, axis):
	'''
	Remove the dataset mean.
	:param features [num_samples,...]
	:param axis the axis to compute mean
	
	'''
	feature_mean=np.mean(features,axis=axis,keepdims=True)
	feature_remove_mean=features-feature_mean
	return feature_remove_mean,feature_mean

def select_balanced_subset(images, labels, use_num_images, use_classes):
	'''
	select equal number of images from each classes
	'''
	# Shuffle
	num_total=images.shape[0]
	shuffle_idx=np.random.permutation(num_total)
	images=images[shuffle_idx]
	labels=labels[shuffle_idx]

	num_class=len(use_classes)
	num_per_class=int(use_num_images/num_class)
	selected_images=np.zeros((use_num_images,images.shape[1],images.shape[2],images.shape[3]))
	selected_labels=np.zeros(use_num_images)
	for i in range(num_class):
		images_in_class=images[labels==i]
		selected_images[i*num_per_class:(i+1)*num_per_class]=images_in_class[:num_per_class]
		selected_labels[i*num_per_class:(i+1)*num_per_class]=np.ones((num_per_class))*i

	# Shuffle again
	shuffle_idx=np.random.permutation(num_per_class*num_class)
	selected_images=selected_images[shuffle_idx]
	selected_labels=selected_labels[shuffle_idx]

	print(selected_images.shape)
	print(selected_labels[0:10])
	plt.figure()
	for i in range(10):
		img = selected_images[i, :, :, 0]
		plt.imshow(img)
		plt.show()
	return selected_images, selected_labels
	# For test
	# print(selected_images.shape)
	# print(selected_labels[0:10])
	# plt.figure()
	# for i in range (10):
	# 	img=selected_images[i,:,:,0]
	# 	plt.imshow(img)
	# 	plt.show()
	return selected_images,selected_labels


'''def laplace_mech(A,sensitivety,epsilon):
	for i in range (len(A)):
		A[i]+=noisycount(sensitivety,epsilon)
	print(len(A))
	return A
def noisycount(sensitivety,epsilon):
	beta = sensitivety / epsilon
	u1 = np.random.random()
	u2 = np.random.random()
	if u1 <= 0.5:
		n_value = -beta * np.log(1. - u2)
	else:
		n_value = beta * np.log(u2)
	print(n_value)
	return n_value'''
def lmatch(a,sensitivty,epsilon1):
	w_conv1Noise=np.random.laplace(0.0,(5*sensitivty)/(epsilon1),25)
	w_conv1Noise=np.reshape(w_conv1Noise,25)
	T=a+w_conv1Noise
	print(w_conv1Noise)
	return T
def lmatch_1(a,sensitivty,epsilon1,value):
	s={}
	for i in range(len(value)):
		s[i]=(value[i]/sum(value))*epsilon1
	r = []
	for j in range(len(s)):
		w = np.random.laplace(0.0, sensitivty/ s[j], 25)
		w=np.reshape(w,25)
		r.append(w)
	u = np.array(r)
	print(u)
	t = u + a
	return t
def lmatch_2(a,sensitivty,epsilon1,value):
	s={}
	value_1=value[::-1]
	for i in range(len(value_1)):
		s[i]=(value_1[i]/sum(value_1))*epsilon1
	r = []
	for j in range(len(s)):
		w = np.random.laplace(0.0, sensitivty/ s[j], 25)
		w=np.reshape(w,25)
		r.append(w)
	u = np.array(r)
	print(u)
	t = u + a
	return t
def find_kernels_pca(samples, num_kernels, energy_percent,epsilon1):
	'''
	Do the PCA based on the provided samples.
	If num_kernels is not set, will use energy_percent.
	If neither is set, will preserve all kernels.

	:param samples: [num_samples, feature_dimension]
	:param num_kernels: num kernels to be preserved
	:param energy_percent: the percent of energy to be preserved
	:return: kernels, sample_mean
	'''

	pca=PCA(n_components=samples.shape[1], svd_solver='full')

	pca.fit(samples)

	# Compute the number of kernels corresponding to preserved energy
	if  energy_percent:
		energy=np.cumsum(pca.explained_variance_ratio_)
		num_components=np.sum(energy<energy_percent)+1
	else:
		num_components=num_kernels
	a=pca.components_[:num_components,:]
	print(a)
	A=np.max(a)
	B=np.min(a)
	sensitivty=(A-B)*5
	print(sensitivty)
	print(num_components)
	value = pca.singular_values_[:num_components]
	sum_value = sum(value)
	#kernels=lmatch(a,sensitivty,epsilon1)
	kernels = lmatch_1(a, sensitivty, epsilon1, value)
	#kernels = lmatch_2(a, sensitivty, epsilon1, value)
	mean=pca.mean_



	print("Num of kernels: %d"%num_components)
	#print("Energy percent: %f"%np.cumsum(pca.explained_variance_ratio_)[num_components-1])
	#print(pca.explained_variance_ratio_)
	#print(value)
	#print(sum(value))
	#print(kernels)
	#print(kernels[0].shape)
	print(A)
	print(B)
	#print(sensitivty)
	#print(a)
	#print(a[0])
	return kernels, mean
def find_kernels_pca_1(samples, num_kernels, energy_percent,epsilon1):

	'''
	Do the PCA based on the provided samples.
	If num_kernels is not set, will use energy_percent.
	If neither is set, will preserve all kernels.

	:param samples: [num_samples, feature_dimension]
	:param num_kernels: num kernels to be preserved
	:param energy_percent: the percent of energy to be preserved
	:return: kernels, sample_mean
	'''
	pca = PCA(n_components=samples.shape[1], svd_solver='full')

	pca.fit(samples)



	# Compute the number of kernels corresponding to preserved energy
	if  energy_percent:
		energy=np.cumsum(pca.explained_variance_ratio_)
		num_components=np.sum(energy<energy_percent)+1
	else:
		num_components = num_kernels


	kernels=pca.components_[:num_components,:]
	mean=pca.mean_
	print("Num of kernels: %d"%num_components)
	#print("Energy percent: %f"%np.cumsum(pca.explained_variance_ratio_)[num_components-1])
	return kernels, mean


def multi_Saab_transform(images, labels, kernel_sizes, num_kernels, energy_percent, use_num_images, use_classes):
	'''
	Do the PCA "training".
	:param images: [num_images, height, width, channel]
	:param labels: [num_images]
	:param kernel_sizes: list, kernel size for each stage,
	       the length defines how many stages conducted
	:param num_kernels: list the number of kernels for each stage,
	       the length should be equal to kernel_sizes.
	:param energy_percent: the energy percent to be kept in all PCA stages.
	       if num_kernels is set, energy_percent will be ignored.
    :param use_num_images: use a subset of train images
    :param use_classes: the classes of train images
    return: pca_params: PCA kernels and mean
    '''

	num_total_images=images.shape[0]
	if use_num_images<num_total_images and use_num_images>0:
		sample_images, selected_labels=select_balanced_subset(images, labels, use_num_images, use_classes)
	else:
		sample_images=images
	# sample_images=images
	num_samples=sample_images.shape[0]
	num_layers=len(kernel_sizes)
	pca_params={}
	pca_params['num_layers']=num_layers
	pca_params['kernel_size']=kernel_sizes

	for i in range(num_layers):
		print('--------stage %d --------'%i)
		epsilon=2
		epsilon1=(5/6)*epsilon
		epsilon2=epsilon-epsilon1
    	# Create patches
		# sample_patches=window_process(sample_images,kernel_sizes[i],kernel_sizes[i]) # nonoverlapping
		sample_patches=window_process(sample_images,kernel_sizes[i],1) # overlapping
		h=sample_patches.shape[1]
		w=sample_patches.shape[2]
    	# Flatten
		sample_patches=sample_patches.reshape([-1, sample_patches.shape[-1]])

    	# Remove feature mean (Set E(X)=0 for each dimension)
		sample_patches_centered, feature_expectation=remove_mean(sample_patches, axis=0)
    	# Remove patch mean
		training_data, dc=remove_mean(sample_patches_centered, axis=1)

		if not num_kernels is None :
			num_kernel=num_kernels[i]
		if i==0:
		  kernels, mean=find_kernels_pca(training_data, num_kernel, energy_percent,epsilon1)
		else :
			kernels, mean = find_kernels_pca_1(training_data, num_kernel, energy_percent,epsilon1)
    	# Add DC kernel
		num_channels=sample_patches.shape[-1]
		print(num_channels)
		if i==0:
		 dc_kernel=1/np.sqrt(num_channels)*np.ones((1,num_channels))
		 p=np.random.laplace(0.0, dc_kernel[0][1]/ epsilon2, 25)
		 print(dc_kernel)
		 print(p)
		 dc_kernel=dc_kernel+p
		else :
			dc_kernel = 1 / np.sqrt(num_channels) * np.ones((1, num_channels))
		kernels=np.concatenate((dc_kernel, kernels), axis=0)

		if i==0:
			# Transform to get data for the next stage
			transformed=np.matmul(sample_patches_centered, np.transpose(kernels))
		else:
	    	# Compute bias term
			bias=LA.norm(sample_patches, axis=1)
			bias=np.max(bias)
			pca_params['Layer_%d/bias'%i]=bias
			# Add bias
			sample_patches_centered_w_bias=sample_patches_centered+1/np.sqrt(num_channels)*bias
			# Transform to get data for the next stage
			transformed=np.matmul(sample_patches_centered_w_bias, np.transpose(kernels))
	    	# Remove bias
			e=np.zeros((1, kernels.shape[0]))
			e[0,0]=1
			transformed-=bias*e
			print(bias)

    	# Reshape: place back as a 4-D feature map
		sample_images=transformed.reshape(num_samples, h, w,-1)

		# Maxpooling
		sample_images=block_reduce(sample_images, (1,2,2,1), np.max)


		print('Sample patches shape after flatten:', sample_patches.shape)
		print('Kernel shape:', kernels.shape)
		print('Transformed shape:', transformed.shape)
		print('Sample images shape:', sample_images.shape)
    	
		pca_params['Layer_%d/feature_expectation'%i]=feature_expectation
		pca_params['Layer_%d/kernel'%i]=kernels
		pca_params['Layer_%d/pca_mean'%i]=mean

	return pca_params

# Initialize
def initialize(sample_images, pca_params):

	num_layers=pca_params['num_layers']
	kernel_sizes=pca_params['kernel_size']

	for i in range(num_layers):
		print('--------stage %d --------'%i)
		# Extract parameters
		feature_expectation=pca_params['Layer_%d/feature_expectation'%i]
		kernels=pca_params['Layer_%d/kernel'%i]
		mean=pca_params['Layer_%d/pca_mean'%i]

    	# Create patches
		#sample_patches = window_process(sample_images, kernel_sizes[i], 5)
		sample_patches=window_process(sample_images,kernel_sizes[i],1) # overlapping
		h=sample_patches.shape[1]
		w=sample_patches.shape[2]
    	# Flatten
		sample_patches=sample_patches.reshape([-1, sample_patches.shape[-1]])
		pca_params['Layer_%d/feature_expectation' % i] = feature_expectation

    	# Remove feature mean (Set E(X)=0 for each dimension)
		sample_patches_centered, feature_expectation=remove_mean(sample_patches, axis=0)
		# sample_patches_centered=sample_patches-feature_expectation
    	
    	# Remove patch mean
		training_data, dc=remove_mean(sample_patches_centered, axis=1)

		num_channels=sample_patches.shape[-1]
		if i==0:
			# Transform to get data for the next stage
			transformed=np.matmul(sample_patches_centered, np.transpose(kernels))
			trans = {}
			trans['transformed'] = transformed
			fw = open('trans.pkl', 'wb')
			pickle.dump(trans, fw)
			fw.close()
		else:
			bias=pca_params['Layer_%d/bias'%i]
			# Add bias
			sample_patches_centered_w_bias=sample_patches_centered+1/np.sqrt(num_channels)*bias
			# Transform to get data for the next stage
			transformed=np.matmul(sample_patches_centered_w_bias, np.transpose(kernels))
	    	# Remove bias
			e=np.zeros((1, kernels.shape[0]))
			e[0,0]=1
			transformed-=bias*e
    	
    	# Reshape: place back as a 4-D feature map
		num_samples=sample_images.shape[0]
		sample_images=transformed.reshape(num_samples, h, w,-1)

		# Maxpooling
		sample_images=block_reduce(sample_images, (1,2,2,1), np.max)

		print('Sample patches shape after flatten:', sample_patches.shape)
		print('Kernel shape:', kernels.shape)
		print('Transformed shape:', transformed.shape)
		print('Sample images shape:', sample_images.shape)
	return sample_images




