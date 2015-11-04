import featurizer
import gen_training_data
import numpy as np
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import SGDClassifier

data = gen_training_data.gen_training_data();
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
