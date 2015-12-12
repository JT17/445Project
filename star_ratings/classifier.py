def classify_reviews():
	import featurizer
	import gen_training_data
	import numpy as np
	from sklearn.naive_bayes import MultinomialNB
	from sklearn.linear_model import SGDClassifier

	data = gen_training_data.gen_data();
	stemmed_data = featurizer.stem(data);
	tfidf= featurizer.tfidf(data);
	clf = MultinomialNB().fit(tfidf['train_tfidf'], data['training_labels']);
	predicted = clf.predict(tfidf['test_tfidf']);
	num_wrong = 0;
	tot = 0;
	for expected, guessed in zip(data['testing_labels'], predicted):
		if(expected-guessed != 0):	
			num_wrong += 1;

	print("num_wrong: %d",num_wrong)

	sgd_clf = SGDClassifier(loss='hinge', penalty='l2', alpha=1e-3, n_iter=5, random_state=42);
	_ = sgd_clf.fit(tfidf['train_tfidf'], data['training_labels']);
	sgd_pred = sgd_clf.predict(tfidf['test_tfidf']);
	print np.mean(sgd_pred == data['testing_labels']);

	stem_tfidf = featurizer.tfidf(stemmed_data);
	_ = sgd_clf.fit(stem_tfidf['train_tfidf'], data['training_labels']);
	sgd_stem_prd = sgd_clf.predict(stem_tfidf['test_tfidf']);
	print np.mean(sgd_stem_prd==data['testing_labels']);

def classify_user_reviews():
	import featurizer
	import gen_training_data
	import numpy as np
	from sklearn.naive_bayes import MultinomailNB
	from sklearn.linear_model import SGDClassifier

	data = gen_training_data.gen_user_data();
	user_classifiers = {};


def user_utility_nUsrClstr(userId, businessId, w1, w2, b):
	#utility function without using user clustering
	import cPickle as pickle
	with open('business_data.p', 'rb') as handle:
		data = pickle.load(handle);

	business_clusters = pickle.load(open('clustered_business.p', "rb"));	
	cluster = business_clusters[businessId];
	pickle.dump(business_clusters, open("clustered_business.p", "wb"));

	user_weights = pickle.load(open("user_weights.p", "rb"));

	average_rating = data[businessId]['stars']; 
	star_prediction = user_weights[userId][cluster - 1]; # should this be -1 or +0?
	

	return w1 * star_prediction + w2 * average_rating + b;

def user_utility_UsrClstr(userId, businessId, w1, w2, b):
	import featurizer
	import cPickle as pickle
	with open('business_data.p', 'rb') as handle:
		data = pickle.load(handle);

	business_clusters = pickle.load(open('clustered_business.p', "rb"));
	cluster = business_clusters[businessId];
	pickle.dump(business_clusters, open("clustered_business.p", "wb"));

	average_rating = data[businessId]['stars'];

	user_weights = pickle.load(open('user_weights.p', 'rb'));
	user_clusters = featurizer.kmeans(user_weights, 32);

	count = 0;
	total = 0:

	for user in user_weights.keys():
		if user_clusters[userId] == user_clusters[user]:
			total += user_weights[user][cluster - 1];
			count += 1;


	star_prediction = total / count;


	return w1 * star_prediction + w2 * average_rating + b;

def error():
	import featurizer
	import cPickle as pickle	

	totalRMSE = 0;
	totalMAE = 0;

	user_weights = pickle.load(open('user_weights.p', 'rb'));
	business_clusters = pickle.load(open('clustered_business.p', 'rb'));

	with open('../../yelp_dataset_challenge_academic_dataset/yelp_academic_dataset_review.json') as yelp_reviews:
		i = 0;
		for review in yelp_reviews:
			if (i > 9000):
				break;
			review_contents = json.loads(review);
			userId = review_contents['user_id'];
			businessId = review_contents['business_id'];
			totalRMSE += (review_contents['stars'] - user_weights[userId][business_clusters[businessId] - 1])**2;
			totalMAE += (review_contents['stars'] - user_weights[userId][business_clusters[businessId] - 1]);

	return {'RMSE':((1 / 9000) * totalRMSE)**.5, 'MAE':((1 / 9000) * totalMAE)**.5};








if __name__ == "__main__":
	classify_reviews();
