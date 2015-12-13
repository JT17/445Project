def gen_all_data():
    import json
    data = {} 
    data_labels = [] 
    partition = []
    with open('../../yelp_dataset_challenge_academic_dataset/yelp_academic_dataset_review.json') as yelp_reviews:
        i = 1;
        key = 0;
        for review in yelp_reviews:
            if (i % 100000 == 0):
                data[key] = partition
                key = key + 1
                partition = []
            review_contents = json.loads(review);
            partition.append(review_contents['text']);
            data_labels.append(review_contents['stars'] );	
            i = i+1;
            print i;
    data[key] = partition
    return {'text':data, 'labels':data_labels}

def tokenize(l):
    
    from nltk.tokenize import RegexpTokenizer
    results = []
    tokenizer = RegexpTokenizer('[a-z]\w+')
    for document in l:
        results.append(tokenizer.tokenize(document.lower()))
    return results

def stem_stopwords(l):

    from nltk.corpus import stopwords
    from nltk.stem.porter import *
    stop_words = stopwords.words('english')
    stemmer = PorterStemmer()
    results = []
    pos_words = load_preprocess('positive_words')
    neg_words = load_preprocess('negative_words')
    for document in l:
        temp_text = []
        for word in document:
            if word not in stop_words:
                stemmed_word = stemmer.stem(word)
                temp_text.append(stemmer.stem(stemmed_word))
                if stemmed_word in pos_words:
                    temp_text.append('GOODREVIEW')
                elif stemmed_word in neg_words:
                    temp_text.append('BADREVIEW')
        results.append(temp_text)
    return results

def preprocess(l):

    print 'tokenizing...\n'
    l = tokenize(l)
    print 'stemming and stopwords...\n'
    l = stem_stopwords(l)

    return l 

def save_preprocess(l, pathname):
    import cPickle
    
    with open(pathname, 'wb') as savefile:
        cPickle.dump(l, savefile)

def load_preprocess(pathname):
    import cPickle

    l = []
    with open(pathname) as savefile:
        l = cPickle.load(savefile)
    return l

def tf(l):
    result = [[]]
    from sklearn.feature_extraction.text import CountVectorizer

    words_to_sentences(l)
    vectorizer = CountVectorizer(min_df = 1, decode_error = 'ignore')
    result = vectorizer.fit_transform(l)

    return result

def gensim_lda(d):
    from gensim import corpora, models
    from gensim.models.ldamodel import LdaModel
    list_doc = []
    for i in range(0,len(d)):
        list_doc = list_doc + d[i]

    dictionary = corpora.Dictionary(list_doc)
    model = LdaModel(num_topics = 20, id2word = dictionary)
    for i in range(0, len(d)):
        print 'Generating corpus and updating model ', i
        corpus = [dictionary.doc2bow(doc) for doc in d[i]]
        model.update(corpus)

    model.save('model_20')
    print model.show_topics(num_topics = 20, num_words = 10)

def main():
#    all_reviews = gen_all_data()
#    reviews_text = all_reviews['text']
#    reviews_labels = all_reviews['labels']
    
#    PREPROCESS
#    for i in range(0, len(reviews_text)):
#        print len(reviews_text[i])
#        print "PARTITION ", i
#        print '\n'
#       part = preprocess(reviews_text[i])
#save_preprocess(part, 'preprocess_codeword' + str(i))

#   TRAINING
    training_data = {}
    for i in range(0, 8):
        print 'loading ', i
        training_data[i] = load_preprocess('preprocess_codeword' + str(i))
    gensim_lda(training_data)
    
#    from gensim import corpora, models
#    from gensim.models.ldamodel import LdaModel
#    model = LdaModel.load('model_test')
#    print model.show_topics(num_topics=10, num_words=10)
    
if __name__ == '__main__':
    main() 
