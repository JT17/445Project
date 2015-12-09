import cPickle as pickle
with open('business_data.p', 'rb') as handle:
	data = pickle.load(handle);

import numpy as np
vals = np.array(data.keys())
print vals.shape
import featurizer
#results = featurizer.kmeans(data,32);
#print results
