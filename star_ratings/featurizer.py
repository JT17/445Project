def stem(input):
	from nltk import PorterStemmer
	stemmer = PorterStemmer();
	stemmed_training_input = [];
	stemmed_testing_input = [];
	for training_example in input['training']:
		word_list = training_example.split();
		stemmed_training_input.append(' '.join([stemmer.stem(word) for word in word_list]))

	for testing_example in input['testing']:
		word_list = testing_example.split();
		stemmed_testing_input.append(' '.join([stemmer.stem(word) for word in word_list]))

	result = {'training':stemmed_training_input, 'training_labels':input['training_labels'], 'testing':stemmed_testing_input, 'testing_labels':input['testing_labels']}
	return result


def tfidf(input):
	from sklearn.feature_extraction.text import CountVectorizer
	from sklearn.feature_extraction.text import TfidfTransformer
	count_vect = CountVectorizer(decode_error='ignore', stop_words='english');
	tfidf_transformer = TfidfTransformer()
	word_counts = count_vect.fit_transform(input['training']);
	train_tfidf = tfidf_transformer.fit_transform(word_counts)

	test_counts = count_vect.transform(input['testing']);
	test_tfidf = tfidf_transformer.transform(test_counts);

	return {'train_tfidf': train_tfidf, 'test_tfidf':test_tfidf}


def kmeans(input, num_clusters):
	from sklearn.cluster import KMeans
	import numpy as np
	from scipy.sparse import csr_matrix
	input_as_np = np.array(input.values());
	dense_vals = np.empty([input_as_np.shape[0], input_as_np[0].shape[1]]);
	index = 0;
	for val in input_as_np:
		dense_vals[index] = val.toarray();
		index = index + 1;
	
	clf = KMeans(n_clusters = num_clusters);
	clustered_input = clf.fit_predict(dense_vals);
	output = {};
	for business_id, cluster in zip(input.keys(), clustered_input):
		output[business_id] = cluster;

	return output
