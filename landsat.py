import pandas as pd
import numpy as np

def load_landsat_data(filename):
	'''
	Utility function to load Landsat dataset.
    
    https://github.com/abarthakur/trepan_python/blob/master/run.py
    
	Landsat dataset : https://archive.ics.uci.edu/ml/datasets/Statlog+(Landsat+Satellite)
	num_classes= 7, but 6th is empty. 
	This functions
	- Reads the data
	- Renames the class 7 to 6
	- Generates one-hot vector labels
	'''

	data = pd.read_csv(filename, sep=r" ", header=None)
	data = data.values

	dataX = np.array(data[:,range(data.shape[1]-1)])
	dataY = np.array(data[np.arange(data.shape[0]),data.shape[1]-1])

	# convert dataY to one-hot, 6 classes
	num_classes = 6
	dataY = np.array([x-2 if x==7 else x-1 for x in dataY]) # re-named class 7 to 6 as class 6 is empty
	dataY_onehot = np.zeros([dataY.shape[0], num_classes])
	dataY_onehot[np.arange(dataY_onehot.shape[0]), dataY] = 1

	return pd.DataFrame(dataX), pd.DataFrame(dataY_onehot)