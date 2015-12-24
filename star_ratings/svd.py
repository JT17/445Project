def review_stats():
	num = 0;
	ave = 0;
	with open('../../yelp_dataset_challenge_academic_dataset/yelp_academic_dataset_review.json') as yelp_reviews:
		for review in yelp_reviews:
			review_contents = json.loads(review);
			num = num + 1;
			ave = ave + review_contents['stars'];
	ave = ave / (1.0 * i);

def gen_business_data_all():
	import json
	
	businesses = []; # id's of all businesses
	users = []; # id's of all users
	
	with open('../../yelp_dataset_challenge_academic_dataset/yelp_academic_dataset_business.json') as yelp_businesses:
		for business in yelp_businesses:
			business_contents = json.loads(business);
			businesses_food.append(business_contents['business_id']);
	businesses.sort(); # all id's in the businesses file should be unique
	
	with open('../../yelp_dataset_challenge_academic_dataset/yelp_academic_dataset_review.json') as yelp_reviews:
		for review in yelp_reviews:
			review_contents = json.loads(review);
			users_food.append(review_contents['user_id']);
	users = list(set(users));
	users.sort();
	
	return businesses, users;

def gen_business_data_food():
	import json
	
	businesses_food = []; # id's of food-related businesses 
	users_food = []; # id's of users who wrote at least one food-related review
	
	with open('../../yelp_dataset_challenge_academic_dataset/yelp_academic_dataset_business.json') as yelp_businesses:
		for business in yelp_businesses:
			business_contents = json.loads(business);
			if ('Restaurants' in business_contents['categories']):
				businesses_food.append(business_contents['business_id']);
	businesses_food.sort();
	
	with open('../../yelp_dataset_challenge_academic_dataset/yelp_academic_dataset_review.json') as yelp_reviews:
		for review in yelp_reviews:
			review_contents = json.loads(review);
			if (review_contents['business_id'] in businesses_food):
				users_food.append(review_contents['user_id']);
	users_food = list(set(users_food));
	users_food.sort();
	
	return businesses_food, users_food;

# input lists of business id's and user id's
def gen_business_ave_stars(businesses, users):
	import json
	
	# dictionary where keys are business id's and values are length-2 lists,
	# with the total number of stars across all reviews for that business and
	# the total number of reviews for that business
	businesses_stars = {};
	for b_id in businesses:
		businesses_stars[b_id] = [0, 0];
	
	with open('../../yelp_dataset_challenge_academic_dataset/yelp_academic_dataset_review.json') as yelp_reviews:
		for review in yelp_reviews:
			review_contents = json.loads(review);
			b_id = review_contents['business_id'];
			u_id = review_contents['user_id'];
			if (b_id in businesses_stars and u_id in users):
				businesses_stars[b_id][0] += review_contents['stars'];
				businesses_stars[b_id][1] += 1;
	
	# change values from length-2 lists to average numbers of stars
	for b_id in businesses_stars:
		if (businesses_stars[b_id][1] > 0):
			businesses_stars[b_id] = businesses_stars[b_id][0] / businesses_stars[b_id][1];
		else:
			businesses_stars[b_id] = 0;
	return businesses_stars;
	


def svd():
	import json
	import copy
	import numpy as np
	import random
	
	# businesses, users = gen_business_data_all();
	businesses, users = gen_business_data_food();
	
	print "Data generated";
	
	num_businesses = len(businesses);
	num_users = len(users);
	for i in range(5 * num_businesses / 6): # using a manageable amount of data
		businesses.pop();
	for i in range(11 * num_users / 12):
		users.pop();
	num_businesses = len(businesses);
	num_users = len(users);
	
	review_matrix = np.zeros((num_businesses, num_users)); # very big
	
	businesses_stars = gen_business_ave_stars(businesses, users);
	
	# fill each row with the average number of stars for that business across all
	# training reviews
	for i in range(num_businesses):
		ave = businesses_stars[businesses[i]];
		if (ave == 0):
			ave = 3;
		for j in range(num_users):
			review_matrix[i][j] = ave;
	
	print "Review matrix allocated";
	
	reviews = {}; # keys are concat(b_id, u_id), values are [b_id, u_id, stars, 0/1]
	
	# fill review_matrix with training data
	with open('../../yelp_dataset_challenge_academic_dataset/yelp_academic_dataset_review.json') as yelp_reviews:
		for review in yelp_reviews:
			review_contents = json.loads(review);
			b_id = review_contents['business_id'];
			u_id = review_contents['user_id'];
			if (b_id in businesses and u_id in users):
				r = random.randint(0,9);
				bu_id = str(b_id) + str(u_id);
				if (r >= 5): # test data
					reviews[bu_id] = [b_id, u_id, review_contents['stars'], 1];
				else: # training data
					reviews[bu_id] = [b_id, u_id, review_contents['stars'], 0];
					b_ind = businesses.index(b_id);
					u_ind = users.index(u_id);
					review_matrix[b_ind][u_ind] = review_contents['stars'];
	
	print "Training data entered";
	
	U, s, V = np.linalg.svd(review_matrix);
	
	S = np.zeros((np.shape(U)[0],np.shape(V)[0]));
	
	#S = np.diag(s); # diagonal matrix whose entries are given by the vector s
	for i in range(len(s)):
		N = 100; # hyperparameter to be tuned: best rank <= 100 approximation
		if (i < N):
			S[i][i] = s[i];
	prediction_matrix = np.dot(U,np.dot(S,V));
	
	print "SVD prediction matrix computed";
	
	# compute RMSE and MAE
	rmse = 0;
	mae = 0;
	count = 0;
	for key in reviews:
		if (reviews[key][3] == 1): # test data
			count = count + 1;
			b_ind = businesses.index(reviews[key][0]);
			u_ind = users.index(reviews[key][1]);
			rmse = rmse + (prediction_matrix[b_ind][u_ind] - reviews[key][2])**2;
			mae = mae + abs(prediction_matrix[b_ind][u_ind] - reviews[key][2]);
	rmse = rmse / (1.0 * count);
	mae = mae / (1.0 * count);
	
	print rmse;
	print mae;
	
	return 0



if __name__ == '__main__':
	svd();
