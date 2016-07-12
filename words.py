from sklearn.feature_extraction.text import CountVectorizer
from utils import DATA_DIR, CHART_DIR
import nltk.stem
import scipy as sp

import sys, os


def dist_raw(v1, v2):
    delta = v1 - v2
    return sp.linalg.norm(delta.toarray())

def dist_norm(v1, v2):
    v1.normalized = v1/sp.linalg.norm(v1.toarray())
    v2.normalized = v2/sp.linalg.norm(v2.toarray())
    delta = v1.normalized - v2.normalized
    return sp.linalg.norm(delta.toarray())

def fingding(new_post_vec, posts, X_train, num_samples):
    best_doc = None
    best_dist = sys.maxsize
    best_i = None
    for i in range(0, num_samples):
        post = posts[i]
        if post == new_post:continue
        post_vec = X_train.getrow(i)
        #d = dist_raw(post_vec, new_post_vec)
        d = dist_norm(post_vec, new_post_vec)
        print("=== Post %i with dist = %.2f: %s" %(i, d, post))
        if d < best_dist:
            best_dist = d
            best_i = i
    print("Best post is %i with dist = %.2f"%(best_i, best_dist))


posts = [open(os.path.join(DATA_DIR, f)).read() for f in os.listdir(DATA_DIR)]
new_post = "imaging databases"

vectorizer = CountVectorizer(min_df=1)
X_train = vectorizer.fit_transform(posts)
num_features = len(X_train.toarray().transpose())
num_samples = len(X_train.toarray())
new_post_vec = vectorizer.transform([new_post])
fingding(new_post_vec, posts, X_train, num_samples)

"""remove stop words"""
print("\n*********remove stop word*********\n")
vectorizer_1 = CountVectorizer(min_df=1, stop_words = 'english') 
X_train_1 = vectorizer_1.fit_transform(posts)
num_features = len(X_train_1.toarray().transpose())
num_samples = len(X_train_1.toarray())
new_post_vec = vectorizer_1.transform([new_post])
fingding(new_post_vec, posts, X_train_1, num_samples)


"""Extending the verctorizer with NLTK's stemmer"""
print("\n*********Extending the verctorizer with NLTK's stemmer*********\n")
english_stemmer = nltk.stem.SnowballStemmer('english')
class StemmedCountVectorizer(CountVectorizer):
    def build_analyzer(self):
        analyzer = super(StemmedCountVectorizer, self).build_analyzer()
        return lambda doc:(english_stemmer.stem(w) for w in analyzer(doc))
                           
vectorizer_2 = StemmedCountVectorizer(min_df=1, stop_words = 'english')
X_train_2 = vectorizer_2.fit_transform(posts)
num_features = len(X_train_2.toarray().transpose())
num_samples = len(X_train_2.toarray())
new_post_vec = vectorizer_2.transform([new_post])
fingding(new_post_vec, posts, X_train_2, num_samples)
