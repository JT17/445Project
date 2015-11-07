def gen_data():
	import json
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
	
	
	return {'training':training, 'training_labels':training_labels, 'testing':testing, 'testing_labels':testing_labels}

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


def cluster_users():
	import json
	training = []
	training_labels = [] 
	testing_category = {}
	
	testing = []
	testing_labels = []
	with open('../../yelp_dataset_challenge_academic_dataset/yelp_academic_dataset_review.json') as yelp_reviews:
		i = 0;
		for review in yelp_reviews:
			review_contents = json.loads(review);
			if(i > 9000):
				if(review_contents['user_id'] in testing):
					testing[review_contents['user_id']].append(review_contents['text']);
					testing_labels[review_contents['user_id']].append(review_contents['stars']);
					testing_category[review_contents['user_id']].append(review_contents['categories'])
				else:
					testing[review_contents['user_id']] = []
					testing_labels[review_contents['user_id']] = []
#testing_category[review_contents['user_id']] = []
					testing[review_contents['user_id']].append(review_contents['user_id']);
					testing_labels[review_contents['user_id']].append(review_contents['user_id']);
#					testing_category[review_contents['user_id']].append(review_contents['user_id']);
			else:
				if(review_contents['user_id'] in training):
					training[review_contents['user_id']].append(review_contents['text']);
					training_labels[review_contents['user_id']].append(review_contents['stars']);
#				training_category[review_contents['user_id']].append(review_contents['categories']);
				else:
					training[review_contents['user_id']] = []
					training_labels[review_contents['user_id']] = []
#training_category[review_contents['user_id']] = []
					training[review_contents['user_id']].append(review_contents['user_id']);
					training_labels[review_contents['user_id']].append(review_contents['user_id']);
#					training_category[review_contents['user_id']].append(review_contents['categories']);
			i = i+1;	
	
#	return {'training':training, 'training_labels':training_labels, 'testing':testing, 'testing_labels':testing_labels, 'training_categories' : training_category, 'testing_categories':testing_category}
	
if __name__ == '__main__':
	parse_business_data();
