from sklearn.feature_extraction.text import CountVectorizer
from utils import DATA_DIR, CHART_DIR
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

posts = [open(os.path.join(DATA_DIR, f)).read() for f in os.listdir(DATA_DIR)]

vectorizer = CountVectorizer(min_df=1)

X_train = vectorizer.fit_transform(posts)

num_features = len(X_train.toarray().transpose())
num_samples = len(X_train.toarray())
new_post = "imaging databases"
new_post_vec = vectorizer.transform([new_post])
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
