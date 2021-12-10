import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
fr=open('pca_params.pkl','rb')
pca_params=pickle.load(fr, encoding='latin1')

fr.close()

i=0
kernels = pca_params['Layer_%d/kernel' % i]
corelation = abs(np.corrcoef(kernels))

sns.heatmap(corelation, cmap='gray',vmax=1,vmin=0,linewidths=0.5,square=[1,1])
plt.show()
