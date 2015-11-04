def gen_training_data():
	import json
	training = []
	training_labels = [] 
	testing = []
	testing_labels = []
	with open('yelp_dataset_challenge_academic_dataset/yelp_academic_dataset_review.json') as yelp_reviews:
		i = 0;
		for review in yelp_reviews:
			review_contents = json.loads(review);
			if(i > 10000):
				testing.append(review_contents['text']);
				testing_labels.append(review_contents['stars'] );	
			else:
				training.append(review_contents['text']);
				training_labels.append(review_contents['stars'] );
			if(i > 10100):
				break;
			i = i+1;
	
	
	return {'training':training, 'training_labels':training_labels, 'testing':testing, 'testing_labels':testing_labels}
