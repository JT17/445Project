import cPickle as pickle
with open('business_data.p', 'rb') as handle:
	data = pickle.load(handle);

import numpy as np
vals = np.array(data.keys())
print vals.shape
import featurizer
print data
results = featurizer.kmeans(data,32);
print len(set(results.values()))
pickle.dump(results, open("clustered_business.p", "wb"))
#print results
user_weights = pickle.load(open('user_weights.p', 'rb'));
user_clusters = featurizer.kmeans(user_weights, 32);
print user_clusters
