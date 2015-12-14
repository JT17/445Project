import cPickle
import json

num_categories = 10
result = {k:[] for k in range(num_categories)} 
categories = {}
with open('clustered_business.p') as savefile:
        categories = cPickle.load(savefile)

business_objects = {}
with open('../../yelp_dataset_challenge_academic_dataset/yelp_academic_dataset_business.json') as yelp_businesses:
        for business in yelp_businesses:
                business_contents = json.loads(business);
                business_objects[business_contents['business_id']] = business_contents['categories'] 

for key in categories:
        result[categories[key]].append(business_objects[key])

print result[8]
        
