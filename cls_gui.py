import pandas as pd
import numpy as np
import copy
env_df = pd.read_csv('./data/environmental_2k.csv')
soc_df = pd.read_csv('./data/social_2k.csv')
gov_df = pd.read_csv('./data/governance_2k.csv')

df = pd.DataFrame(copy.deepcopy(env_df['text']))
label = env_df.iloc[:,2] + soc_df.iloc[:,2]+ gov_df.iloc[:,2]
label = [0 if c == 0 else 1 for c in label]
df['label'] = label

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import RidgeClassifier
from sklearn.metrics import f1_score

tfidf = TfidfVectorizer(ngram_range=(1, 3), max_features=300)
X = tfidf.fit_transform(df['text'])

from sklearn.model_selection import train_test_split
X_train, X_test,y_train,y_test = train_test_split(X,df['label'],train_size=0.7,stratify=df['label'])

import sklearn.naive_bayes as nb
model_nb = nb.MultinomialNB()      # 创建朴素贝叶斯模型
model_nb.fit(X_train,y_train)
pred = model_nb.predict(X_test)
print(f1_score(y_test,pred , average='macro'))
# 0.6258

clf_rig = RidgeClassifier()
clf_rig.fit(X_train, y_train)
val_pred = clf_rig.predict(X_test)
print(f1_score(y_test, val_pred, average='macro'))
# 0.6666

### KNN Classifier
from sklearn.neighbors import KNeighborsClassifier

clf = KNeighborsClassifier()
clf.fit(X_train, y_train)
val_pred = clf.predict(X_test)
print(f1_score(y_test, val_pred, average='macro'))
# 0.6320

### Logistic Regression Classifier
from sklearn.linear_model import LogisticRegression

clf_lr = LogisticRegression(penalty='l2',C=100)
clf_lr.fit(X_train, y_train)
val_pred = clf_lr.predict(X_test)
pred = clf_lr.predict_proba(X_test)
pred = [1 if p[1] > 0.6895 else 0 for p in pred]
print(f1_score(y_test, pred, average='macro'))
print("--")
print(f1_score(y_test, val_pred, average='macro'))

# 0.6940

### Random Forest Classifier
from sklearn.ensemble import RandomForestClassifier

clf = RandomForestClassifier(n_estimators=8)
clf.fit(X_train, y_train)
val_pred = clf.predict(X_test)
print(f1_score(y_test, val_pred, average='macro'))
# 0.6255

### Decision Tree Classifier
from sklearn import tree

clf = tree.DecisionTreeClassifier()
clf.fit(X_train, y_train)
val_pred = clf.predict(X_test)
print(f1_score(y_test, val_pred, average='macro'))

# 0.6065

### SVM Classifier
from sklearn.svm import SVC

clf = SVC(kernel='rbf', probability=True)
clf.fit(X_train, y_train)
val_pred = clf.predict(X_test)
print(f1_score(y_test, val_pred, average='macro'))

# 0.5557

## Save model
import pickle
# pkl_filename = "cls_model.pkl"
# with open(pkl_filename, 'wb') as file:
#     pickle.dump(clf_rig, file)
# tfidftransformer_path = 'tfidftransformer.pkl'
# with open(tfidftransformer_path, 'wb') as fw:
#     pickle.dump(tfidf, fw)
# pkl_filename = "clf_lr_model.pkl"
# with open(pkl_filename, 'wb') as file:
#     pickle.dump(clf_lr, file)