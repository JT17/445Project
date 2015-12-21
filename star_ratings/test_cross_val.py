import cPickle as pickle
import numpy as np
import classifier
import featurizer
import gen_training_data
from scipy import sparse

numBusinessClusters = 10;
numUserClusters = 10;
business_data = pickle.load(open("business_data.p", "rb"))
business_clusters = featurizer.kmeans(business_data,numBusinessClusters);
pickle.dump(business_clusters['data_clusters'], open('clustered_business.p', 'wb'))
user_data = gen_training_data.cluster_users(numUserClusters,5);
user_clusters = featurizer.kmeans(user_data['training'], numUserClusters);
pickle.dump(user_clusters['data_clusters'], open('clustered_user.p', 'wb'))
input_as_np = np.array(user_data['validation'].values());
dense_vals = np.empty([input_as_np.shape[0], input_as_np[0].shape[1]]);
index = 0;
for val in input_as_np:
	dense_vals[index] = val.toarray();
	index = index + 1;
validation_clusters = user_clusters['cluster_info'].predict(dense_vals);
#print validation_clusters;
output = {};
for v_id, cluster in zip(user_data['validation'].keys(),validation_clusters):
	output[v_id] = cluster;
error = classifier.errorV(numUserClusters,numBusinessClusters,output);
print error
pickle.dump(error, open('error.p', 'wb'))
