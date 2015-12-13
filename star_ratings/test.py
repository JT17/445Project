import cPickle as pickle
import numpy as np
import classifier
import featurizer
import gen_training_data
from scipy import sparse
business_data = pickle.load(open("business_data.p", "rb"))
business_clusters = featurizer.kmeans(business_data,32);
pickle.dump(business_clusters, open('clustered_business.p', 'wb'))
user_data = gen_training_data.cluster_users(32,5);
user_clusters = featurizer.kmeans(user_data, 32);
pickle.dump(user_clusters, open('clustered_user.p', 'wb'))
error = classifier.error();
print error
pickle.dump(error, open('error.p', 'wb'))
#print results
#user_weights = pickle.load(open('user_weights.p', 'rb'));
#user_clusters = featurizer.kmeans(user_weights, 32);
#pickle.dump(user_clusters, open('user_clusters.p', 'wb'))
#users = user_clusters.keys();
#businesses = business_clusters.keys();
#user_ratings = {};
#predictions = [];
#w1 = 1;
#w2 = 1;
#b = 1;
#for user in users:
#	predictions = [];
#	for business in businesses:
#		predictions.append(user_utility_UsrClstr(user, business, w1, w2, b));
#	
#	predictions = sparse.csr_matrix(predictions);
#	user_ratings[user] = predictions;
#
#

