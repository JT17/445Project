from __future__ import division
def gen_data():
	import json
	import cPickle as pickle
	training = []
	training_labels = [] 
	testing = []
	testing_labels = []
	with open('../../yelp_dataset_challenge_academic_dataset/yelp_academic_dataset_review.json') as yelp_reviews:
		i = 0;
		for review in yelp_reviews:
			review_contents = json.loads(review);
			if(i > 9000):
				testing.append(review_contents['text']);
				testing_labels.append(review_contents['stars'] );	
			else:
				training.append(review_contents['text']);
				training_labels.append(review_contents['stars'] );
			i = i+1;
	return_struct = {'training':training, 'training_labels':training_labels, 'testing':testing, 'testing_labels':testing_labels}
	with open('label_data.p', 'wb') as fp:
		pickle.dump(return_struct,fp);	
	return return_struct;


def parse_business_data():
	import json
	import copy
	from scipy import sparse
	import cPickle as pickle
	categories = {};
	business_categories = {};
	with open('../../yelp_dataset_challenge_academic_dataset/yelp_academic_dataset_business.json') as yelp_businesses:
			for business in yelp_businesses:
				business_contents = json.loads(business);
				for category in business_contents['categories']:
					categories[category] = 0;
	
	with open('../../yelp_dataset_challenge_academic_dataset/yelp_academic_dataset_business.json') as yelp_businesses:
			for business in yelp_businesses:
				business_contents = json.loads(business);
				business_categories_list = copy.deepcopy(categories);
				for category in business_contents['categories']:
					business_categories_list[category] = 1;
				business_categories[business_contents['business_id']] = sparse.csr_matrix(business_categories_list.values());
	
	with open('business_data.p', 'wb') as fp:
		pickle.dump(business_categories, fp);
	return business_categories;	


def cluster_users(num_clusters, useful_threshold):
	import json
	import cPickle as pickle
	import numpy as np
	import hashlib
	from scipy import sparse
	training_clusters = {} 
	training_num = {} 
	testing_category = {}
	
	testing_clusters ={} 
	testing_num = {} 
	business_clusters = pickle.load(open('clustered_business.p', "rb"));
	num_businesses = len(set(business_clusters.values()))
	num_train_users = 0;
	with open('../../yelp_dataset_challenge_academic_dataset/yelp_academic_dataset_review.json') as yelp_reviews:
		i = 0;
		for review in yelp_reviews:
			review_contents = json.loads(review);
			if(i > 1500000):
				if(review_contents['user_id'] in testing_clusters):
					testing_clusters[review_contents['user_id']][business_clusters[review_contents['business_id']]] += review_contents['stars'];
					testing_num[review_contents['user_id']][business_clusters[review_contents['business_id']]] += 1 
				else:
					testing_clusters[review_contents['user_id']] = [0] * num_businesses 
					testing_num[review_contents['user_id']] = [0] * num_businesses
					testing_clusters[review_contents['user_id']][business_clusters[review_contents['business_id']]] += review_contents['stars'];
					testing_num[review_contents['user_id']][business_clusters[review_contents['business_id']]] += 1 
				if(review_contents['votes']['useful'] > useful_threshold):
					for val in xrange(review_contents['votes']['useful'] - useful_threshold):
						testing_clusters[review_contents['user_id']][business_clusters[review_contents['business_id']]] += review_contents['stars'];
						testing_num[review_contents['user_id']][business_clusters[review_contents['business_id']]] += 1 

			else:
				if(review_contents['user_id'] in training_clusters):
					training_clusters[review_contents['user_id']][business_clusters[review_contents['business_id']]] += review_contents['stars'];
					training_num[review_contents['user_id']][business_clusters[review_contents['business_id']]] += 1 
				else:
					num_train_users += 1;
					training_clusters[review_contents['user_id']] = [0] * num_businesses
					training_num[review_contents['user_id']] = [0] * num_businesses
					training_clusters[review_contents['user_id']][business_clusters[review_contents['business_id']]] += review_contents['stars'];
					training_num[review_contents['user_id']][business_clusters[review_contents['business_id']]] += 1 
				
				if(review_contents['votes']['useful'] > useful_threshold):
					for val in xrange(review_contents['votes']['useful'] - useful_threshold):
						training_clusters[review_contents['user_id']][business_clusters[review_contents['business_id']]] += review_contents['stars'];
						training_num[review_contents['user_id']][business_clusters[review_contents['business_id']]] += 1 
			i = i+1;	
	
	elite_users = {}
	with open('../../yelp_dataset_challenge_academic_dataset/yelp_academic_dataset_user.json') as yelp_users:
		for user in yelp_users:
			user_info = json.loads(user);
			if(user_info['elite'] > 0):
				elite_users[user_info['user_id']] = 1;
	
	training_weights = {} 
	i = 0;
	for user, business_categories, num_revs in zip(training_clusters.keys(), training_clusters.values(), training_num.values()):
		j = 0;
		vals = [];
		for weight, num in zip(business_categories, num_revs):
			if(num != 0):
				vals.append(weight/float(num));
				# training_weights[user] = weight/float(num);
			else:
				vals.append(0);
			j += 1;
		training_weights[user] = sparse.csr_matrix(vals);
		if(user in elite_users):
			training_weights[user+'aaaaaaa'] = sparse.csr_matrix(vals)
		i += 1;

	pickle.dump(training_weights, open("user_weights.p", "wb"))
	return training_weights;	
#	return {'training':training, 'training_labels':training_labels, 'testing':testing, 'testing_labels':testing_labels, 'training_categories' : training_category, 'testing_categories':testing_category}
	
if __name__ == '__main__':
#parse_business_data();
	cluster_users(32, 5);
	import cPickle as pickle
	vals = pickle.load(open("user_weights.p", "rb"));
	print vals
